import streamlit as st
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Load your saved model
@st.cache_resource
def load_c_model():
    model = load_model('garbage_classifier_model.keras') # or your saved filename
    return model

model = load_c_model()

# Page steup
st.set_page_config(
    page_title="Waste Classification App",
    layout="wide",
    page_icon="♻️",
    initial_sidebar_state='expanded')


st.sidebar.subheader("Image Classification",divider="grey",icon=":material/compost:")

#st.sidebar.subheader("AI Prediction",divider="grey",icon=":material/apartment:")

# 2. Define class labels in the EXACT order of your training generator
#st.image("Glass Bottle for prediction in CNN Model.png",use_container_width=True)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ RecycleVision: Waste Classification App")
st.image("Gemini_Generated Grabage.png",use_container_width="stretch",width=3000)

from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        "Image Classification",
        ["Home","Classify the Image","Contact Us"],
        icons=["house-fill","image","people"],
        default_index=0)

# Below are the icons available in icons package
#---------------------------------------------------
# house,house-fill,person,person-fill,people,people-fill,search,gear,gear-fill,table,database,
# calendar,calendar-event,trophy,award,star,star-fill,heart,heart-fill
# book,clipboard-data,bar-chart,bar-chart-fill,graph-up,graph-up-arrow,
# graph-down,pie-chart,globe,camera,image,file-earmark,folder,download,
# upload,play-circle,pause-circle,check-circle,x-circle,info-circl


#----------------------------------------------------------------------------
#----------------------------  Home Page ------------------------------------
#----------------------------------------------------------------------------
if selected == "Home":
    st.write("""♻️ Welcome to RecycleVision

                Smart Waste Classification Using Deep Learning

                Garbage Image Classification Using Deep Learning.docx
                Welcome to your intelligent recycling assistant! This application leverages state-of-the-art Computer Vision and Deep Learning (MobileNetV2) to help automate waste sorting, reduce environmental impact, and promote sustainable recycling habits.   
                Garbage Image Classification Using Deep Learning.docx

                🚀 What You Can Do Here
                🔍 Classify Waste Instantly: Navigate to the "Classify" page from the sidebar, upload a photo of a garbage item, and let the AI instantly identify its category with a confidence score.   
                Garbage Image Classification Using Deep Learning.docx

                📊 Explore Our Categories: Our model is trained to recognize 6 distinct types of everyday waste items:

                📦 Cardboard

                🍾 Glass

                🥫 Metal

                📄 Paper

                🥤 Plastic

                🗑️ Trash / General Waste

                💡 Why Waste Classification Matters
                Proper waste segregation is the backbone of effective recycling and municipal waste management. By automatically sorting recyclables from general waste, we can:   
                Garbage Image Classification Using Deep Learning.docx

                Reduce Landfill Waste: Ensure recyclable materials like plastic, glass, and paper actually get recycled.

                Save Time & Labor: Assist municipal facilities and smart recycling bins in sorting waste efficiently.   
                Garbage Image Classification Using Deep Learning.docx

                Promote Education: Teach proper environmental habits through visual, AI-powered tools.   
                Garbage Image Classification Using Deep Learning.docx

                🛠️ Project Tech Stack
                Deep Learning Framework: TensorFlow / Keras

                Transfer Learning Architecture: MobileNetV2

                Frontend & Deployment: Streamlit   
                Garbage Image Classification Using Deep Learning.docx

                📌 How to Get Started
                Use the sidebar on the left to switch to the Classify the Image page, snap or upload a photo of your waste item, and see the model in action! """)


#----------------------------------------------------------------------------
#--------------------------  Classification Page ----------------------------
#----------------------------------------------------------------------------

if selected == "Classify the Image":
    st.write("Upload an image of garbage to classify it automatically!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open image using PIL
        img_display = Image.open(uploaded_file)
        img_to_display = img_display.resize((224, 224))
        st.image(img_to_display, caption='Uploaded Image', use_container_width=False)
        
        # --- CRITICAL PREPROCESSING STEPS ---
        # Convert image to RGB (in case it's RGBA/PNG with transparency)
        img = img_display.convert('RGB')
        
        # Resize image to match model input shape (224x224)
        img = img.resize((224, 224))
        
        # Convert PIL image to a numpy array
        img_array = image.img_to_array(img)
        
        # Expand dimensions to create a batch of 1 -> Shape: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values from [0, 255] to [0, 1]
        img_array = img_array / 255.0
        
        # Make prediction safely
        with st.spinner('Classifying...'): # Displays a temporary loading spinner with a status message while the model runs prediction
            predictions = model.predict(img_array) # Passes the preprocessed image array through the trained deep learning model to get probability scores across all classes
            predicted_class_idx = np.argmax(predictions[0]) # Finds the index position of the class with the highest probability score from the prediction array
            predicted_class = class_labels[predicted_class_idx] # Maps the winning index back to its corresponding human-readable category name (e.g., 'plastic', 'metal')
            confidence = np.max(predictions[0]) * 100 # Extracts the highest probability value and multiplies by 100 to convert it into a percentage confidence score

        @st.dialog("🎞️ Image Classification Results") # This line runs along with function
        def show_prediction():

            st.subheader("📷 Image Classification Results")

            st.metric("Uplaoded Image is",
                        f" {predicted_class.upper()}")
            st.divider()

            col7, col8 = st.columns(2)

            with col7:
                st.metric("Uplaoded Image is", f" {predicted_class.upper()} ")

            with col8:
                st.metric("Confidence Score", f"{confidence:.2f}%")

        # Calling Both the functions
        show_prediction()

        # Display results
        st.success(f"**Predicted Category:** {predicted_class.upper()}")
        st.info(f"**Confidence Score:** {confidence:.2f}%")