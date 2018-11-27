import cv2
import numpy as np
import os
import re
 
path = '/home/skalwar/ultrasonic_image/train/'
size=572
size1=572
def pre_processing():
	pass


def post_processing():
	pass

if __name__=="__main__":
	list_img = os.listdir(path)
	#print("No of Images:-",len(list_img))
	for i in list_img:
		if re.search(r"[0-9]_[0-9].tif",i):
			img = cv2.imread(path+i)
			img = cv2.resize(img,(size,size),cv2.INTER_AREA)
			path1 = "/home/skalwar/unet_implementation/train_x/"+i
			cv2.imwrite(path1,img)

	for i in list_img:
		if re.search(r"[0-9]_[0-9]_mask.tif",i):
			img = cv2.imread(path+i)
			img = cv2.resize(img,(size1,size1),cv2.INTER_AREA)
			path1 = "/home/skalwar/unet_implementation/train_y/"+i.replace('_mask.tif','.tif')
			cv2.imwrite(path1,img)

	list_train_x=os.listdir("/home/skalwar/unet_implementation/train_x/")
	list_train_y=os.listdir("/home/skalwar/unet_implementation/train_y/")
	len_train_x = len(list_train_x)
	len_train_y = len(list_train_y)

	X_train = np.zeros((len_train_x,1,size,size))
	Y_train = np.zeros((len_train_y,1,size1,size1))
	print("X_train shape:-",X_train.shape)
	print("Y_train shape:-",Y_train.shape)
	print("list train x",list_train_x)
	print("list train y",list_train_y)	
	i=0
	for j in list_train_x:
		path1 = "/home/skalwar/unet_implementation/train_x/"+j
		X_train[i,0,:,:]= cv2.imread(path1,0)
		#print(X_train[i,:,:])
		i=i+1
	np.save("X_train.npy",X_train)
	
	i=0
	for j in list_train_y:
		path1 = "/home/skalwar/unet_implementation/train_y/"+j
		Y_train[i,:,:]= cv2.imread(path1,0)
		i=i+1
	np.save("Y_train.npy",Y_train)

	x=np.load("X_train.npy")
	y=np.load("Y_train.npy")
	temp = x.copy()
	x1=temp[0,:,:]
	x1[np.where(y[0,:,:]==0)]=0
	cv2.imshow("temp",x1)
	cv2.imshow("image",x[0,:,:])
	cv2.imshow("output_mask",y[0,:,:])
	cv2.waitKey(0)
