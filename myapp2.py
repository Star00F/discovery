import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from PIL import ImageOps
import cv2
import io
import time

# Set page title and favicon
st.set_page_config(
    page_title="Image Processing App",
    page_icon="🌟"
)
st.header("Welcome to the Image Processing App")
st.write("The animals that the AI is trained on are in this list:")
st.markdown("Papegaaien, Dolfijnen, Schildpadden, Zebra's, Mollen, Paarden, Tijgers, Giraffen, Eenhoorns, Ezels, Leeuwen, Draken, Honden, Capibara's, Kittens, olifante, Wolven, Jaguars, Gorilla's, Bonobo's, Lynxen, Korhoenders, Sneeuwluipaarden, Nijlpaarden, Zeesterren, Flamingo's, Ooievaars, Zeepaarden, Zeehonden, Vossen en Mensen")

# Load model and class names
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# Option to choose between file upload and camera
option = st.radio("Choose an option:", ("Upload File", "Use Camera"))

if option == "Upload File":
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the uploaded image and make predictions
        pil_image = Image.open(uploaded_file)
        data = preprocess_image(pil_image)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display the uploaded image
        st.image(pil_image, channels="RGB", use_column_width=True)

        # Display the result
        st.text(f"Now it is a {class_name[2:]} with confidence: {confidence_score:.2f}")

elif option == "Use Camera":
    # Webcam feed
    cap = cv2.VideoCapture(0)

    # Display the webcam feed and result on Streamlit
    webcam_feed = st.empty()
    result_text = st.empty()
    timer_text = st.empty()

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to PIL Image
        pil_frame = Image.fromarray(frame_rgb)

        # Preprocess the image and make predictions
        data = preprocess_image(pil_frame)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display the live webcam feed on Streamlit
        webcam_feed.image(frame_rgb, channels="RGB", use_column_width=True)

        # Display the result under the live webcam feed
        result_text.text(f"Now it is a {class_name[2:]} with confidence: {confidence_score:.2f}")

        # Display the time elapsed
        elapsed_time = time.time() - start_time
        timer_text.text(f"Time Elapsed: {elapsed_time:.2f} seconds")

        # Wait for 1 second
        time.sleep(1)

    # Release the webcam feed at the end
    cap.release()
