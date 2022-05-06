import streamlit as st
from fastai.widgets import *
from fastai.vision import *
from PIL import Image

def load_image(image_file):
	img = Image.open(image_file)
	return img

st.set_page_config(layout="wide")
st.header("Image Classifier: Tank, Truck, or Plane")

st.markdown(
    "This is a simple image classifier using a convolutional neural network (CNN) for image classification. "
    "The model used in training is the resenet34 which is a residual network 34 layer neural net. "
    # "The training data is available for confirmation as well as an URL option to classify your own images. "
)

@st.cache(allow_output_mutation=True)
def long_running_function():
    learn = load_learner('.','trainedModel.pkl')
    return learn

learn = long_running_function()
printformat = dict({"trucks": "a truck", "tanks": "a tank", "planes": "a plane"})


st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

    # To View Uploaded Image
    st.image(load_image(image_file),width=400)

    # Open the image for the model to look at
    img = open_image(image_file)
    
    # Run the loaded model on the image
    pred_class,pred_idx,outputs = learn.predict(img)

    # Print out what the model returns
    st.markdown(f"I believe you have uploaded {printformat[str(pred_class)]} ")

