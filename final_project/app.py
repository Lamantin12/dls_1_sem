import streamlit as st
from model import usage

st.set_page_config(layout='wide')

@st.cache_resource
def load_models_(path):
    model_A_B, model_B_A = usage.load_models(path)
    st.success("Loaded Model")
    return model_A_B, model_B_A

def app(model_A_B, model_B_A):
    st.header("DLS FINAL PROJECT") 
    in_image = st.file_uploader("load image here", [".jpg"])
    confirm_button = st.button("confirm")
    inp, out = st.columns(2)
    if in_image is None:
        return
    print(type(in_image))
    print(type(in_image.read())) 
    if confirm_button:  
        inp.image(in_image)
        out.image(usage.process_image_from_image(model_A_B, model_B_A, in_image, "B2A"))



model_A_B, model_B_A = load_models_(r"./weights/")
app(model_A_B, model_B_A)
