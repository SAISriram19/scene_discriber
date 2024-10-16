# Scene Discriber

Description
Scene Discriber is a Python application that detects objects in images and generates descriptive captions using deep learning models.  
It combines a pre-trained ResNet model for object detection and the BLIP model for image captioning.

Features  
- Object detection using ResNet50
- Scene description generation using the BLIP model
- User-friendly interface to upload images for analysis

**Installation**  
Clone the repository:
bash
git clone https://github.com/SAISriram19/scene_discriber
cd scene_discriber
conda create -n scene_discriber python=3.8
conda activate scene_discriber
pip install -r requirements.txt
streamlit run app.py
