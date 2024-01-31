import streamlit as st
from fastai.vision.all import *
st.write("Pizza Vs Not Pizza Classifier")
st.text("Builtport stream by Theo Kang")

def label_func(f): return f[0].isupper()


model = load_learner('my_model (1).pkl')

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = model.predict(img)
    likelihood_is_pizza = outputs[1].item()
    if likelihood_is_pizza < 0.7:
        return "Pizza"
    else:
        return "Not Pizza"

st.title("Pizza vs. Not-Pizza Classifier")
st.write("Upload an image, and I'll tell you whether it's pizza or not-pizza.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = predict(uploaded_file)
        st.write(prediction)

st.text("Built with Streamlit and Fastai")
