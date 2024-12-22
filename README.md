# MArine-Reasearch-Project
an exciting university project focused on marine wildlife interaction using a machine vision system. The system involves an already built underwater device with multiple interactive components that respond to specific triggers.

A peer has developed a YOLOv7-based machine vision model that accurately identifies marine animals based on their facial features. I’m building a mobile app using React and Vite to integrate this vision model, enabling the app to identify subjects using the phone’s camera.

I’m seeking an machine vision developer to:

Load the existing YOLOv7 model onto the React and Vite app I’ve developed.

Test the model’s performance and functionality within the app.

Test and compare other machine vision models (e.g., Apple’s CoreML models) to identify the best-performing solution.

Document the entire process, including steps taken, comparisons, and findings.

Provide clear explanations so I can understand how the models function and how they were integrated.

I’ll provide a full dataset of training images and videos for you to work with, as well as extensive documentation on the current YOLO model. I can provide additional images upon request if needed for better model performance.

The ideal candidate would have expertise in machine vision, Python, and React, as well as experience with YOLOv7, CoreML, TensorFlow Lite, or similar frameworks.

You need to be able to get to work straight away and complete the project with a quick turnaround.
----------
To build the application you're describing, we need to break the project into the following key steps:

    YOLOv7 Model Integration into React/Vite App: Integrate the YOLOv7-based model for marine wildlife identification into the React/Vite application.
    Model Testing & Comparison: Test the performance of the YOLOv7 model within the app and compare it with other machine vision models (such as Apple’s CoreML or TensorFlow Lite).
    Documentation: Provide clear documentation of the process, testing, comparison of models, and implementation.

Here's a Python-based solution that walks through how to load YOLOv7 for inference, convert it into a model format suitable for mobile (using TensorFlow Lite or CoreML), and integrate it into a React Native app.
Step 1: YOLOv7 Model Inference in Python (Backend)

We will start by loading the YOLOv7 model in Python and testing it on a sample image. You can use PyTorch to load the YOLOv7 model and make predictions.
Install Required Libraries

pip install torch torchvision opencv-python

Load YOLOv7 Model for Inference

import torch
import cv2
import numpy as np

# Load YOLOv7 model (Assuming the model is saved as a `.pt` file)
model = torch.hub.load('ultralytics/yolov7', 'yolov7', pretrained=True)

# Function to process the image and make predictions
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the image to the input size expected by YOLOv7 (640x640)
    img_resized = cv2.resize(img, (640, 640))

    # Convert the image to a tensor and normalize it
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 640, 640]

    # Perform inference (get the model’s predictions)
    results = model(img_tensor)

    # Extract bounding boxes and labels from results
    boxes = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
    labels = results.names
    predictions = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = labels[int(cls)]
        predictions.append({
            'label': label,
            'confidence': conf,
            'bounding_box': [x1, y1, x2, y2]
        })
    
    return predictions

# Example usage:
image_path = 'marine_animal_image.jpg'
predictions = process_image(image_path)
print(predictions)

Step 2: Convert YOLOv7 Model to TensorFlow Lite or CoreML

Since React Native doesn't directly support PyTorch models, we need to convert the model into a format that is compatible with mobile. This can be done by converting the YOLOv7 model to either TensorFlow Lite (for Android) or CoreML (for iOS).
Convert PyTorch Model to ONNX (Intermediary Step)

import torch.onnx

# Export YOLOv7 model to ONNX format
dummy_input = torch.randn(1, 3, 640, 640)  # Dummy input for export
onnx_path = "yolov7.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)

Convert ONNX to TensorFlow (for TensorFlow Lite)

You can use the onnx-tf library to convert the ONNX model to a TensorFlow model and then to TensorFlow Lite.

pip install onnx-tf

import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load("yolov7.onnx")
tf_model = prepare(onnx_model)

# Convert to TensorFlow Lite
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

# Save the converted model
with open("yolov7_model.tflite", "wb") as f:
    f.write(tflite_model)

Convert ONNX to CoreML (for iOS)

To convert to CoreML (for iOS), you can use the onnx-coreml tool:

pip install onnx-coreml

from onnx_coreml import convert

# Convert ONNX model to CoreML format
coreml_model = convert(model="yolov7.onnx")
coreml_model.save("yolov7_model.mlmodel")

Step 3: React Native Mobile App (Frontend)

Now, let’s create a mobile app using React Native and TensorFlow Lite or CoreML. This app will use the phone’s camera to capture an image, send it to the backend (for YOLOv7 inference), or directly run inference on the mobile device using the converted models.
Install Dependencies

npx react-native init MarineWildlifeApp
cd MarineWildlifeApp
npm install @tensorflow/tfjs @tensorflow/tfjs-react-native

Camera Setup in React Native

You can use the react-native-camera package to allow the user to take pictures of marine wildlife.

npm install react-native-camera

TensorFlow Lite Inference in React Native

In your app, you’ll load the TensorFlow Lite model and use it to make predictions.

import React, { useState, useEffect } from 'react';
import { View, Text, Button, Image, TouchableOpacity } from 'react-native';
import { Camera } from 'react-native-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as tflite from '@tensorflow/tfjs-lite';

const MarineApp = () => {
  const [model, setModel] = useState(null);
  const [imageUri, setImageUri] = useState(null);
  const [diagnosis, setDiagnosis] = useState('');

  useEffect(() => {
    async function loadModel() {
      await tf.ready();
      const tfliteModel = await tflite.loadTFLiteModel('path_to_your_tflite_model.tflite');
      setModel(tfliteModel);
    }

    loadModel();
  }, []);

  const captureImage = async () => {
    // Capture image from camera
    const options = { quality: 0.5, base64: true };
    const data = await camera.takePictureAsync(options);
    setImageUri(data.uri);
    classifyImage(data.uri);
  };

  const classifyImage = async (uri) => {
    // Perform inference on captured image
    const imageTensor = tf.browser.fromPixels(uri);
    const predictions = await model.predict(imageTensor);
    setDiagnosis(predictions);
  };

  return (
    <View>
      <Camera ref={ref => {camera = ref}} />
      <TouchableOpacity onPress={captureImage}>
        <Text>Capture Image</Text>
      </TouchableOpacity>
      {diagnosis && <Text>Diagnosis: {diagnosis}</Text>}
    </View>
  );
};

export default MarineApp;

Step 4: Testing and Optimization

Test the system with different images of marine animals to ensure the YOLOv7 model performs well. If performance is suboptimal, you might need to adjust the model’s hyperparameters or explore other models.
Documentation

Finally, document the entire process:

    Steps Taken: From loading YOLOv7 in Python, converting the model to TensorFlow Lite or CoreML, and integrating it into the mobile app.
    Comparison of Models: Include the performance of YOLOv7, CoreML, TensorFlow Lite, and other machine vision models for the specific task.
    Challenges Faced: Explain any issues related to model performance or integration, and how you solved them.

Conclusion

This solution outlines how to:

    Load YOLOv7 and other models for marine wildlife identification.
    Convert the model to a mobile-friendly format like TensorFlow Lite or CoreML.
    Build a React Native app that integrates with the machine vision models for real-time predictions.

By following this structure, you can create a powerful marine wildlife identification app that leverages state-of-the-art machine vision models.
