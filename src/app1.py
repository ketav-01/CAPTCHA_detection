import streamlit as st
import cv2
import numpy as np
import os
import torch
from model import SpinalVGG
from segmentation import Segmentation

# Function to load the model and predict captcha
def predict_captcha(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpinalVGG().to(device)
    model.load_state_dict(torch.load("WEIGHT\\model.h5", map_location=device))

    path_list = Segmentation(image_path)
    classes = "A,C,E,H,N,P,R,S,T,U,W,X,Y,Z,b,d,n,q,t,1,2,3,4,5,6,7"
    index = "10,12,14,17,23,25,27,28,29,30,32,33,34,35,37,38,43,44,46,62,63,64,65,66,67,68"
    values = classes.split(',')
    keys = index.split(',')
    output_dict = dict(zip(keys, values))
    ans_list = []
    for j in path_list:
        img = cv2.imread(j, 0)
        if img is None:
            print("Error: Unable to load image")
        else:
            print("Image loaded successfully")
        
        if img is not None:
            resized_image = cv2.resize(img, (28, 28))
        else:
            print("Error: Unable to load or decode the image.")
        img = np.reshape(resized_image, (1, 1, 28, 28))
        img = torch.tensor(img, dtype=torch.float)
        img = img.to(device)
        model.eval()
        x = model(img)
        _, predicted = torch.max(x.data, 1)
        ans_list.append(output_dict[f'{predicted.item()}'])

    return ''.join(map(str, ans_list))

# Streamlit app
def main():
    st.title("Captcha Recognition App")
    
    # Input field for image path
    image_path = st.text_input("Enter the image path:")

    if st.button("Predict"):
        if image_path:
            prediction = predict_captcha(image_path)
            st.write("Predicted Captcha:", prediction)
        else:
            st.write("Please enter an image path.")

if __name__ == '__main__':
    main()
