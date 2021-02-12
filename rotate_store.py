import os
import numpy as np
import pandas as pd
import cv2
import random
import splitfolders

current_path = os.getcwd()
image_dir=os.path.join(current_path, "Front")
# current_path="/home/mansi/Dossier_1_Lac_Data"

resized_folder=os.path.join(current_path,'Resized')
resized_orign1=os.path.join(resized_folder,'-1')
resized_orig0=os.path.join(resized_folder,'0')
resized_orig1=os.path.join(resized_folder,'1')
resized_orig2=os.path.join(resized_folder,'2')

# df = pd.DataFrame(columns=['img','rotation'])
possible_rotations={90:cv2.ROTATE_90_COUNTERCLOCKWISE ,180:cv2.ROTATE_180 ,-90: cv2.ROTATE_90_CLOCKWISE, -1: -1}		#cv2.ROTATE_90_CLOCKWISE=0, cv2.ROTATE_180=1, cv2.ROTATE_90_COUNTERCLOCKWISE=2
keys=list(possible_rotations.keys())
random.seed(10)

if not os.path.exists(resized_folder):
	os.mkdir(resized_folder)

if not os.path.exists(resized_orign1):
	os.mkdir(resized_orign1)

if not os.path.exists(resized_orig0):
	os.mkdir(resized_orig0)

if not os.path.exists(resized_orig1):
	os.mkdir(resized_orig1)

if not os.path.exists(resized_orig2):
	os.mkdir(resized_orig2)


dropped_from_rotate_store=list()
rotation_list=list()
resize=224
i=0
for image in os.listdir(image_dir):
	# print(image)
	image_name, extension = image.split(".")
	if("_" in image):
			l = image_name.split("_")[1]
			image_id = l.split("-")[0]

	else:
		image_id=image

	image_path=os.path.join(image_dir,image)
	img_raw=cv2.imread(image_path, cv2.IMREAD_COLOR)

	if img_raw is None:
		print('None')
		dropped_from_rotate_store.append(image)
		continue

	# cv2.imshow('Window',img_raw)
	# cv2.waitKey(0)

	# print('Prev: ',img_raw.shape)

	try:
		img_raw=cv2.resize(img_raw, (resize, resize),interpolation=cv2.INTER_AREA)

	except:
		print('Couldn\'t resize')
		dropped_from_rotate_store.append(image)


	rotate_by=random.choice(keys)

	if(rotate_by!=-1):
		img_raw=cv2.rotate(img_raw, possible_rotations[rotate_by])

	# print('After: ',img_raw.shape)

	# cv2.imshow('Window',img_raw)
	# cv2.waitKey(0)
	
	write_to=os.path.join(resized_folder,str(possible_rotations[rotate_by]))
	if img_raw.shape[0]==224:
		print(img_raw.shape)

		if not os.path.exists(os.path.join(write_to,image)):
			cv2.imwrite(os.path.join(write_to,image),img_raw)
			rotation_list.append(possible_rotations[rotate_by])
		i=i+1

	else:
		print(img_raw.shape[0])
		dropped_from_rotate_store.append(image)
		print('Dropping')
		print(img_raw.shape)
		continue
	# df.loc[i]={'img':img_raw, 'rotation':possible_rotations[rotate_by]}
	# i=i+1

with open('dropped_from_rotate_store.txt', 'w') as f:
	for item in dropped_from_rotate_store:
		f.write("%s\n" % item)


splitfolders.ratio(resized_folder, output="split_data", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values
