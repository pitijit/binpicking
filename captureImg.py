import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Create a folder to save images
save_folder = "captured_images/calibrate"
os.makedirs(save_folder, exist_ok=True)

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream only the color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

print("Press 'C' to capture an image. Press 'Q' to exit.")

try:
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Display the RGB image
        cv2.imshow('RGB Image', color_image)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Generate a timestamp for the filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_folder, f"image_{timestamp}.jpg")
            
            # Save the image
            cv2.imwrite(filename, color_image)
            print(f"Image saved: {filename}")
        
        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    
    # Close OpenCV windows
    cv2.destroyAllWindows()
