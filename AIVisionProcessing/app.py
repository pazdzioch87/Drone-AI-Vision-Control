import cv2
import ultralytics
from flask import Flask
import subprocess
ultralytics.checks()

from IPython import display
display.clear_output()
from ultralytics import YOLO

from IPython.display import display

def create_app():
    app = Flask(__name__)

    try:
        # Create a Popen instance
        process = subprocess.Popen(['dir'], stdout=subprocess.PIPE)

        # Communicate will return stdout, stderr
        out, err = process.communicate()

        # If you want to get the output as a string, you might need to decode it
        out_decoded = out.decode('utf-8')
        print('dir output:', out_decoded)

        # run_result = subprocess.run(['python3', 'run.py', '--source', 'rtmp://172.19.161.123:1935/stream/hello', '--weights', 'yolov5s.pt'], capture_output=True, text=True)
        run_result = subprocess.run(['python3', 'run.py', '--source', '0', '--weights', 'yolov8s.pt', '--show-preview', 'True'], capture_output=True, text=True)

        # Accessing the output
        print('Python script output:', run_result.stdout)
    except Exception as e:
        print("Error occurred:", e)

    @app.route('/')
    def hello_world():
        return 'Hello, World!'

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=6100, debug=True)

