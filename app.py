import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load your pre-trained model
model = load_model("cnnDrowsiness.h5")

# Define classes
classes = [0, 1, 2, 3]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(145, 145))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(img_array):
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return classes[class_index]

def main():
    st.title("Drowsiness Detection App :sleepy:")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Preprocess and classify the uploaded image
        img_array = preprocess_image(uploaded_file)
        class_index = classify_image(img_array)

        st.write("Prediction Class Index: ", class_index)

        # Map class index to class label
        prediction = classes[class_index]

        st.write("Prediction: ", prediction)

        # Display result
        if class_index in [0, 2]:
            st.warning("Drowsy")
        else:
            st.success("Not Drowsy")

if __name__ == "__main__":
    main()
