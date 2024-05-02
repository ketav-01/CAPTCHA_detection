import cv2
import os
import pandas as pd
import numpy as np

df = pd.read_csv('DATA\\trained_combined_data.csv')
df.head()

for i in range(10):
    df = df[df['45'] != i]

# List of values to filter out from column '41'
remove_values = [11, 13, 15, 16, 18, 19, 20, 21, 22, 24, 26, 31, 36, 39, 40, 41, 42, 45]
for value in remove_values:
    df = df[df['45'] != value]

# Ensure '41' column is still present in the DataFrame
if '41' in df.columns:
    # Perform operations on DataFrame
    unique_count = len(df['45'].unique())
    
    label = df['45']
    df.drop(['45'], axis=1, inplace=True)
    df = 255 - df
    df = pd.concat([label, df], axis=1)

df.to_csv("DATA\\letter-dataset.csv",index=False, header=None)

images = []
for num,i in enumerate(os.listdir('DATA\\emoji data Augmented')):
    print(num,i)
    s = 'DATA\\emoji data Augmented\\'
    s += i
    for _,j in enumerate(os.listdir(s)):
        path = s+'\\'+j
        image = []
        img = cv2.imread(path, 0)
        img = np.ravel(img)
        label = 62+num
        img = np.insert(img, 0, label, axis=0)
#         print(img)
        images.append(img)

images = np.array(images)
df = pd.DataFrame(images)
# save the dataframe as a csv file
df.to_csv("DATA\\emoji-dataset.csv",index=False, header=None)

data = data2 = ""

with open("DATA\\letter-dataset.csv") as fp:
    data = fp.read()
  
with open("DATA\\emoji-dataset.csv") as fp:
    data2 = fp.read()

data += "\n"
data += data2
  
with open ('DATA\\combined-dataset.csv', 'w') as fp:
    fp.write(data)