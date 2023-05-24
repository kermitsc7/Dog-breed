import streamlit as st
from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
from PIL import Image

# Define model
def initialize_model_resnext101_64x4d(num_classes):
    model = torchvision.models.resnext101_64x4d()
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model

escut = Image.open('./images/escut_3r_matcad.jpg')
logo = Image.open('./images/logo.png')


st.set_page_config(
    page_title="Dog Breed Classification",
    page_icon=escut,
    layout="wide",
)


col1, mid, col2 = st.columns([1,3,20])
with col1:
    st.image(logo, width=200)

with col2:
    st.markdown('<h1 style="color:White;">Dog Breed classification model 🐶</h1>', unsafe_allow_html=True)



st.markdown('<h2 style="color:gray;">The model classifies dog image into 120 different breeds</h2>', unsafe_allow_html=True)

st.info("IMPORTANT! only upload images with .jpg extension", icon="ℹ️")
#s'ha de cambiar el upload la manera de cridarlo
file_up = st.file_uploader("Upload an image", type = ["jpg"])


def predict(image):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # carreguem el model

    model = initialize_model_resnext101_64x4d(120)
    device = torch.device('cpu')
    model.load_state_dict(torch.load("./ResNext101_64x4d_PreTrained.pth", device))
    model.eval()
    
   
    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image).convert('RGB')
    batch_t = torch.unsqueeze(transform(img), 0)
    #Evaluem en model amb la imatge i mirem els resultats
    model.eval()
    out = model(batch_t)

    with open('datos.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, width=300)


    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    j = 0
    if(labels[0][1] < 5):
            st.error('This is probably not a Dog ')
            st.write("")
            st.write("**Try with another image...**")
    else:
        for i in labels:
                if (j == 0):
                     j +=1
                     st.balloons()
                     name =  i[0].replace('_', ' ')
                     st.header("**The breed dog classification is:**")
                     st.success(name.upper())
                     st.write("With a Score: " , i[1])
                     st.header('')
                     st.write('**Other classifications:**')
                else:  
                    name = i[0].replace('_', ' ')
                    st.write("**Name:**",name.upper() , "    ", ", **Score:** ", i[1])


    import time

