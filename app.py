from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

config_file =  'C:/Users/yerra/Downloads/OBJECT DETECTION/CODING/back_end/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'C:/Users/yerra/Downloads/OBJECT DETECTION/CODING/back_end/frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

class_labels = []
file_name = 'C:/Users/yerra/Downloads/OBJECT DETECTION/CODING/back_end/labels.txt'
with open(file_name, 'rt') as fpt:
    class_labels = fpt.read().rstrip('\n').split('\n')

def perform_object_detection(image_path):
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    img = cv2.imread(image_path)
    classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

    for ClassInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img, boxes, (225, 0, 0), 1)
        cv2.putText(img, class_labels[ClassInd - 1], (boxes[0] + 4, boxes[1] + 20),
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=1)

    result_path = "static/result_image.jpg"
    cv2.imwrite(result_path, img)

    return result_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # Save the uploaded image
            file_path = "static/uploads/uploaded_image.jpg"
            file.save(file_path)

            # Perform object detection
            result_image_path = perform_object_detection(file_path)

            # Display the results on the web page
            return render_template('index.html', image_path=file_path, result_image_path=result_image_path)

    return render_template('index.html', image_path=None, result_image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
