from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
import cv2
import numpy as np
import os
from PIL import Image

datagen = ImageDataGenerator(        
        rotation_range = 30,
        zoom_range = [0.8,1.05],
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True)

for j in os.listdir('DATA\\emoji data'):
    image_directory = f'DATA\\emoji data\\{j}'
    augmented_directory = f'emoji data Augmented\\{j}'
    SIZE = 28
    dataset = []
    my_images = os.listdir(image_directory)
    for i, image_name in enumerate(my_images):    
        if (image_name.split('.')[1] == 'jpg'):        
            image = io.imread(os.path.join(image_directory, image_name))        
            image = Image.fromarray(image, 'RGB')        
            # image = image.resize((SIZE,SIZE)) 
            open_cv_image = np.array(image)
            kernel = np.ones((5,5), np.uint8)
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            
            img_erosion = cv2.erode(open_cv_image, kernel, iterations=2)
            img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
            image = cv2.resize(img_dilation, (28,28), interpolation = cv2.INTER_AREA)
            # (thresh, blackAndWhiteImage) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            dataset.append(np.array(image))
            
    x = np.array(dataset)
    i = 0

    if not os.path.exists(augmented_directory):
        os.makedirs(augmented_directory)
    
    for batch in datagen.flow(x, batch_size=32,
                            save_to_dir= f'emoji data Augmented\\{j}',
                            save_prefix='dr',
                            save_format='jpg'):    
        i += 1    
        if i > 200:        
            break