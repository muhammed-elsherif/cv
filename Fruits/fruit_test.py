import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Object Detection",
    page_icon="👁‍🗨",
    initial_sidebar_state="expanded"
)

# Streamlit UI
st.title("Object Detection")
st.markdown('This is an application for object detection using CNN and YOLO')
st.caption("Upload an image and click the 'Analyse Image' button to detect components in the image.")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "mp4", "webp"])

# Sidebar
st.sidebar.header("Machine Learning Model")
model_type = st.sidebar.radio("Select Model", ("YOLO", "Classic classification CNN", "Bound box CNN", "Bounded label CNN"))
confidence_threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.5, 0.05
)

if False:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        model_path = "best_roboflow_yolo.pt"
        model_path = "best_kaggle_yolo.pt"

        model = YOLO(model_path)

        results = model.predict(frame, conf=0.6)
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR")

if uploaded_file is not None:
    file_type = uploaded_file.type

    # If it's an image
    if file_type in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if st.button("Analyse Image"):
            if model_type == "YOLO":
                # model_path = "best_roboflow_yolo.pt"
                model_path = "best_kaggle_yolo.pt"

                model = YOLO(model_path)
                results = model.predict(image, conf=confidence_threshold)
                processed_frame = results[0].plot()
                st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Detected Components - YOLO")

            elif model_type == "Classic classification CNN":
                model_path = "fruit_detection_classification.h5"
                model = load_model(model_path, compile=False)

                image_resized = cv2.resize(image, (256, 256))
                image_resized = image_resized / 255.0
                image_resized = np.expand_dims(image_resized, axis=0)

                preds = model.predict(image_resized)
                class_index = np.argmax(preds)
                confidence = np.max(preds)

                class_labels = ['apple', 'banana', 'mixed', 'orange']
                predicted_class = class_labels[class_index]

                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{predicted_class} ({confidence:.2f})")

            elif model_type == "Bound box CNN":
                model_path = "fruit_detection_bbox.h5"
                model = load_model(model_path, compile=False)

                image_resized = cv2.resize(image, (256, 256))
                image_resized = image_resized / 255.0  # Normalize pixel values to [0, 1]
                image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
                
                preds = model.predict(image_resized)
                h, w = image.shape[:2]

                # Denormalize coordinates
                px, py, pw, ph = preds[0]
                px, py, pw, ph = int(px * w), int(py * h), int(pw * w), int(ph * h)

                # Convert to rectangle coordinates
                pred_start = (px - pw // 2, py - ph // 2)
                pred_end = (px + pw // 2, py + ph // 2)

                # img = (image * 255).astype(np.uint8)

                img = cv2.rectangle(image, pred_start, pred_end, (0, 0, 255), 2)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
            elif model_type == "Bounded label CNN":
                model_path = "fruit_detection_bbox_label.h5"
                model = load_model(model_path, compile=False)

                image_resized = cv2.resize(image, (256, 256))
                image_resized = image_resized / 255.0  # Normalize pixel values to [0, 1]
                image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension

                pred_boxes, pred_class_probs = model.predict(image_resized)
                pred_classes = np.argmax(pred_class_probs, axis=0)
                pred_classes = np.argmax(pred_class_probs[0])        # Get class index
                confidence = np.max(pred_class_probs[0])             # Confidence score

                h, w = image.shape[:2]

                px, py, pw, ph = pred_boxes[0]

                px, py, pw, ph = int(px * w), int(py * h), int(pw * w), int(ph * h)

                pred_start = (px - pw // 2, py - ph // 2)
                pred_end = (px + pw // 2, py + ph // 2)

                img = (image * 255).astype(np.uint8)

                CLASS_DICT = {'apple': 0, 'banana': 1, 'mixed': 2, 'orange': 3}
                class_name = list(CLASS_DICT.keys())[pred_classes]

                pred_label = list(CLASS_DICT.keys())[pred_classes]

                img = cv2.rectangle(image, pred_start, pred_end, (0, 0, 255), 2)
                cv2.putText(img, f"{class_name} ({confidence:.2f})", pred_start, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{class_name} ({confidence:.2f})")