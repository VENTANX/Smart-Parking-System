import carla
import numpy as np
import time
from ultralytics import YOLO
import math

def draw_bounding_box(world, bbox, rotation, color=carla.Color(0, 255, 0), life_time=1.0):
    """Draw a box in the world with given rotation and lifetime."""
    # Ensure rotation is a carla.Rotation object
    if not isinstance(rotation, carla.Rotation):
        rotation = carla.Rotation()
    world.debug.draw_box(bbox, rotation, thickness=0.05, color=color, life_time=life_time)

def carla_image_to_numpy(image):
    """Convert a CARLA raw image to a numpy array in RGB format."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format
    array = array[:, :, :3]  # drop alpha channel
    array = array[:, :, ::-1]  # BGR to RGB
    return array

def box_center_xyxy(box):
    """Calculate center (x,y) of bounding box [x1,y1,x2,y2]."""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center

def estimate_parking_spots_between_cars(car_boxes, min_spot_width=50):
    """
    Estimate intervals between detected cars bounding boxes in image X coordinates,
    return intervals larger than min_spot_width as parking spots.
    car_boxes: list of bounding boxes [x1,y1,x2,y2]
    min_spot_width: minimum width in pixels to consider as a parking spot
    """
    # Sort cars by left x coordinate left to right
    sorted_boxes = sorted(car_boxes, key=lambda b: b[0])
    parking_spots = []
    # Check gaps between adjacent cars
    for i in range(len(sorted_boxes) - 1):
        current_right = sorted_boxes[i][2]
        next_left = sorted_boxes[i+1][0]
        gap_width = next_left - current_right
        if gap_width >= min_spot_width:
            parking_spots.append( (current_right, next_left) )
    return parking_spots

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get spectator (your free-flying camera)
    spectator = world.get_spectator()

    # Camera sensor blueprint
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')

    # Attach camera to spectator (moves with your view)
    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=spectator)

    model = YOLO('yolo12n.pt')  # Make sure the model file is correct and accessible

    detected_cars_prev = 0
    detected_parking_spots_prev = 0

    def process_image(image):
        nonlocal detected_cars_prev, detected_parking_spots_prev
        img = carla_image_to_numpy(image)

        results = model(img)

        cars_boxes = []
        detected_cars = 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # xyxy in pixel coordinates
            classes = result.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                cls_int = int(cls)
                if cls_int == 2:  # car class
                    cars_boxes.append(box)
                    detected_cars += 1

        if detected_cars != detected_cars_prev:
            print(f"Detected cars: {detected_cars}")
            detected_cars_prev = detected_cars

        image_width = 640
        image_height = 480
        fov = 90
        focal_length = (image_width / 2) / math.tan(math.radians(fov / 2))

        spec_transform = spectator.get_transform()
        spec_loc = spec_transform.location
        spec_rot = spec_transform.rotation
        forward_vector = spec_rot.get_forward_vector()
        right_vector = spec_rot.get_right_vector()
        up_vector = spec_rot.get_up_vector()

        # Clear all previous debug overlays
        # To avoid erasing all (including simulator debug info), do not clear fully
        # Instead, just draw boxes repeatedly with lifetime

        # Draw cars bounding boxes
        for box in cars_boxes:
            x_center, y_center = box_center_xyxy(box)
            x_norm = (x_center - image_width / 2) / (image_width / 2)
            y_norm = (image_height / 2 - y_center) / (image_height / 2)
            box_height = box[3] - box[1]
            depth = 20.0 * (image_height / box_height) if box_height > 0 else 20.0
            depth = min(max(depth, 5.0), 50.0)

            world_loc = spec_loc + forward_vector * depth + right_vector * (x_norm * depth) + up_vector * (y_norm * depth*0.5)

            bbox_extent = carla.Vector3D(2.25, 1.0, 1.5)  # half extents for a car approx
            bbox = carla.BoundingBox(world_loc, bbox_extent)
            # Use zero rotation because box orientation unknown
            zero_rotation = carla.Rotation()
            draw_bounding_box(world, bbox, zero_rotation, color=carla.Color(0,255,0), life_time=1.0)

        # Estimate parking spots
        parking_spots = estimate_parking_spots_between_cars(cars_boxes, min_spot_width=60)
        if len(parking_spots) != detected_parking_spots_prev:
            print(f"Estimated parking spots: {len(parking_spots)}")
            detected_parking_spots_prev = len(parking_spots)

        for left_x, right_x in parking_spots:
            gap_center_x = (left_x + right_x) / 2
            x_norm = (gap_center_x - image_width / 2) / (image_width / 2)
            y_norm = 0

            # Estimate depth between nearest cars or default
            depth = 20.0  # default if no depth info
            world_loc = spec_loc + forward_vector * depth + right_vector * (x_norm * depth) + up_vector * (y_norm * depth*0.5)

            # Draw translucent yellow box for parking spot
            gap_pixel_width = right_x - left_x
            width_meters = gap_pixel_width / image_width * depth * 0.8
            bbox_extent = carla.Vector3D(width_meters, 1.2, 0.2)
            bbox = carla.BoundingBox(world_loc, bbox_extent)
            zero_rotation = carla.Rotation()
            draw_bounding_box(world, bbox, zero_rotation, color=carla.Color(255, 255, 0, 100), life_time=1.0)

    # Test: draw a static box at spectator to verify drawing works
    def draw_test_box():
        loc = spectator.get_transform().location
        bbox_extent = carla.Vector3D(2, 2, 2)
        bbox = carla.BoundingBox(loc, bbox_extent)
        zero_rotation = carla.Rotation()
        world.debug.draw_box(bbox, zero_rotation, thickness=0.1, color=carla.Color(255, 0, 0), life_time=5.0)
        print("Drew test box at spectator location.")

    draw_test_box()

    camera.listen(process_image)

    print("Press Ctrl+C to quit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        camera.destroy()

if __name__ == "__main__":
    main()

