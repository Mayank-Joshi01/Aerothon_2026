import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

# Initialize the Web Server
app = Flask(__name__)

# Step 1: Initialize the main camera object
picam2 = Picamera2()

# Step 2: LOAD THE .RPK FILE
model_path = '/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'
imx500 = IMX500(model_path)
imx500.show_network_fw_progress_bar()

# Step 3: Configure the camera resolution and start it
config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()

LABELS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 14: "bird", 61: "chair", 83: "clock"}


def generate_frames():
    """This function runs in a loop, grabbing frames and sending them to the website."""
    while True:
        request = picam2.capture_request()

        # STREAM 1: THE VIDEO FRAME
        image_frame = request.make_array("main")
        hud_frame = image_frame.copy()

        # STREAM 2: THE RAW AI TENSOR METADATA
        # Using the updated lower-level methods you provided
        metadata = request.get_metadata()
        outputs = imx500.get_outputs(metadata)

        # Process the streams and draw boxes
        if outputs is not None:
            # SSD MobileNetv2 Tensor Output Structure:
            # outputs[0] = Bounding Boxes [y_min, x_min, y_max, x_max]
            # outputs[1] = Confidence Scores
            # outputs[2] = Class IDs
            boxes = outputs[0]
            scores = outputs[1]
            class_ids = outputs[2]

            # Sometimes the IMX500 wraps these in an extra batch array (e.g. shape [1, 100])
            # This safely flattens them so we can loop through them smoothly
            if len(class_ids.shape) > 1:
                boxes = boxes[0]
                scores = scores[0]
                class_ids = class_ids[0]

            for i in range(len(class_ids)):
                conf = float(scores[i])

                if conf > 0.5:  # Only draw if we are 50% sure
                    label_id = int(class_ids[i])
                    label_name = LABELS.get(label_id, f"ID:{label_id}")

                    # Extract the box coordinates (Raw MobileNet format is usually y, x, y, x)
                    box = boxes[i]
                    y_min_norm, x_min_norm, y_max_norm, x_max_norm = box[0], box[1], box[2], box[3]

                    # Convert normalized box coordinates (0.0 to 1.0) to actual pixels
                    height, width, _ = hud_frame.shape
                    x_min = int(x_min_norm * width)
                    y_min = int(y_min_norm * height)
                    x_max = int(x_max_norm * width)
                    y_max = int(y_max_norm * height)

                    # Draw the Box and Text
                    box_color = (0, 255, 0)
                    cv2.rectangle(hud_frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    label_text = f"{label_name} {int(conf * 100)}%"
                    cv2.putText(hud_frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Convert the RGB array to BGR so OpenCV encodes the colors correctly for the web
        hud_frame_bgr = cv2.cvtColor(hud_frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', hud_frame_bgr)
        frame_bytes = buffer.tobytes()

        # Yield the frame to the web server
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        request.release()


@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 JARVIS HUD IS LIVE!")
    print("Open Google Chrome on your laptop and go to:")
    print("http://YOUR_PI_IP_ADDRESS:5000")
    print("Press Ctrl+C in this terminal to stop.")
    print("=" * 50 + "\n")

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping camera...")
        picam2.stop()