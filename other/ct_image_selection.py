
#%% Import libraries and specify directories
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
img_dir = "dataset/ct_image_data/data/image/"
label_dir = "dataset/ct_image_data/data/label/"
table_data_dir = "dataset/ct_image_data/"
imgs_names = os.listdir(img_dir)
label_names = os.listdir(label_dir)
table_name = "hemorrhage_diagnosis_raw_ct.csv"
#%% data cleaning 

diagnosis_raw = pd.read_csv(table_data_dir + table_name)
diagnosis_raw["DirtyImage"] = 0
dirty_data_index = []
sensor_data_index = []
i = 0

img_boarder = np.full((512, 10, 3), (5, 200, 200), dtype = np.uint8)
pbar = tqdm(total=len(imgs_names), desc="Processing images", unit="image")

while i < len(imgs_names):
    img_path = os.path.join(img_dir, imgs_names[i])
    img = cv2.imread(img_path)
    
    label_path = os.path.join(label_dir, label_names[i])
    labels = cv2.imread(label_path)
    
    black_prop_i = np.sum(img == 0) / img.size
    if black_prop_i > 0.9:
        if i not in dirty_data_index:
            dirty_data_index.append(i)
        i+=1
        pbar.update(1)
    elif black_prop_i >= 0.8:
        while True:
            
            two_images = np.hstack((img, img_boarder, labels))
            cv2.imshow("Image #{} for review".format(i), two_images)

            key = cv2.waitKey(0)
            time.sleep(0.2)
            print("black prop: {}".format(black_prop_i))
            if key == ord("0"):
                sensor_data_index.append(i)
                i+=1
                pbar.update(1)
                break
            elif key == ord("1"):
                if i not in dirty_data_index:
                    dirty_data_index.append(i)
                sensor_data_index.append(i)
                print("Image #{} is marked as dirty.".format(i))
                print(dirty_data_index)
                i+=1
                pbar.update(1)
                break
            elif key == ord("9"):
                if len(sensor_data_index) == 0:
                    print("No image to go back to.")
                    continue
                else:
                    i = sensor_data_index[-1]
                    sensor_data_index = sensor_data_index[:-1]
                    dirty_data_index = dirty_data_index[:-1]
                    break
            else:
                print("Invalid key pressed. Please press '1', '0' or '9'.")
                continue
        cv2.destroyAllWindows()
    else:
        i+=1
        pbar.update(1)

diagnosis_raw.loc[dirty_data_index, "DirtyImage"] = 1
diagnosis_raw.to_csv(table_data_dir+"hemorrhage_diagnosis_ct_dirty_labelled.csv", index=False)
# %% double check
dirty_table_name = "hemorrhage_diagnosis_ct_dirty_labelled.csv"
diagnosis_dirty_labelled = pd.read_csv(table_data_dir + dirty_table_name)
dirty_records = diagnosis_dirty_labelled[diagnosis_dirty_labelled["DirtyImage"] == 1]

for i in dirty_records.index:
    img_path = os.path.join(img_dir, imgs_names[i])
    img = cv2.imread(img_path)
    
    label_path = os.path.join(label_dir, label_names[i])
    labels = cv2.imread(label_path)
    
    two_images = np.hstack((img, img_boarder, labels))
    cv2.imshow("Image #{} for review".format(i), two_images)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    time.sleep(0.2)
    print("Image #{} is marked as dirty.".format(i))


#%% save clean data to a new folder

clean_img_dir = "dataset/ct_image_data/data/image_clean/"
clean_label_dir = "dataset/ct_image_data/data/label_clean/"
clean_table_name = "hemorrhage_diagnosis_ct_clean.csv"

dirty_table = pd.read_csv(table_data_dir + dirty_table_name)
clean_table = dirty_table[dirty_table["DirtyImage"] == 0]
clean_table.drop(columns=["DirtyImage"], inplace=True)
clean_table.to_csv(table_data_dir + clean_table_name, index=True)

#%% save clean images
os.makedirs(clean_img_dir, exist_ok=True)
os.makedirs(clean_label_dir, exist_ok=True)

for i in range(len(imgs_names)):
    if i not in dirty_data_index:
        img_path = os.path.join(img_dir, imgs_names[i])
        img = cv2.imread(img_path)
        label_path = os.path.join(label_dir, label_names[i])
        labels = cv2.imread(label_path)
        cv2.imwrite(clean_img_dir + imgs_names[i], img)
        cv2.imwrite(clean_label_dir + label_names[i], labels)


# %%
