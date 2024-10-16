import streamlit as st
from main import SceneSpeak

scene_speak = SceneSpeak()

st.title("Scene Speak")
st.write("Upload an image to detect objects and generate a description.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = uploaded_file.read()
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    if st.button("Describe Scene"):
        description = scene_speak.describe_scene(uploaded_file)
        st.write(description)
