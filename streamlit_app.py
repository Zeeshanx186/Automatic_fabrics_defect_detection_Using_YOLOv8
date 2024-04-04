from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image

import numpy as np

modelpath = r"train7/weights/last.pt"
model = YOLO(modelpath)

st.title('Predict Fabrics Defects')

# Function to capture image from camera
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return ret, frame

# Function to perform prediction
def predict_defect(image):
    result = model(image)
    if isinstance(result, tuple):
        predictions = result[:3]  # Extracting relevant outputs from tuple
        names = predictions[0].names
        probability = predictions[0].probs.data.numpy()
        prediction = np.argmax(probability)
        return names[prediction]
    elif isinstance(result, list):
        predictions = result[0]
        names = predictions.names
        probability = predictions.probs.data.numpy()
        prediction = np.argmax(probability)
        return names[prediction]
    else:
        raise ValueError("Unexpected output format from YOLO model")

# Main Streamlit app
def main():
    st.sidebar.title('Options')
    use_camera = st.sidebar.checkbox('Use Camera', True)
    capture_video = st.sidebar.checkbox('Capture Video', False)

    if use_camera:
        st.subheader('Live Camera Feed')
        capture_button = st.button("Capture Image")
        if capture_button:
            ret, frame = capture_frame()
            if ret:
                st.image(image=frame, channels='RGB')
                names = predict_defect(frame)
                st.write(names)
                if 'good' in names:
                    st.write('Good Image')
                elif 'hole' in names:
                    st.write('Hole defect in Image')
                elif 'objects' in names:
                    st.write('Objects defect in Image')
                elif 'oil' in names:
                    st.write('Oil spot defect in Image')
                elif 'thread' in names:
                    st.write('Thread error defect in Image')
                else:
                    st.write('Unknown defect')

    if capture_video:
        st.subheader('Live Video Feed')
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels='RGB')
                names = predict_defect(frame)
                st.write(names)
                if 'good' in names:
                    st.write('Good Image')
                elif 'hole' in names:
                    st.write('Hole defect in Image')
                elif 'objects' in names:
                    st.write('Objects defect in Image')
                elif 'oil' in names:
                    st.write('Oil spot defect in Image')
                elif 'thread' in names:
                    st.write('Thread error defect in Image')
                else:
                    st.write('Unknown defect')
            else:
                st.write('Error: Unable to capture frame from the camera.')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    st.sidebar.markdown('---')

    st.sidebar.subheader('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        names = predict_defect(image)
        st.write(names)
        if 'good' in names:
            st.write('Good Image')
        elif 'hole' in names:
            st.write('Hole defect in Image')
        elif 'objects' in names:
            st.write('Objects defect in Image')
        elif 'oil' in names:
            st.write('Oil spot defect in Image')
        elif 'thread' in names:
            st.write('Thread error defect in Image')
        else:
            st.write('Unknown defect')

if __name__ == "__main__":
    main()
