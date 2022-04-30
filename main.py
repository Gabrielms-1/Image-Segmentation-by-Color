import image_processing
import glob
import cv2
import os
import shutil
import matplotlib.pyplot as plt

num_clusters = 8
image_list = glob.glob('data/tomato_dataset/*.png', recursive=True)

shutil.rmtree('data/tomato_dataset/processed')
os.mkdir('data/tomato_dataset/processed')

for image in image_list:
    image_ = cv2.imread(image)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    plt.imshow(image_)
    plt.show()
    print(' - Processing', image.split('/')[-1])
    processed_image = image_processing.CloudMaskCreator(image_, num_clusters).clustered_image

    output = image.replace('tomato_dataset', 'tomato_dataset/processed')
    #cv2.imwrite(output, processed_image)
