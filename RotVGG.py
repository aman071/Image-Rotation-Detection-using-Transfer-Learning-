# -*- coding: UTF-8 -*-

# CUDA_VISIBLE_DEVICES=0 nohup python rotate_store.py > output.txt &
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.python.client import device_lib
import tensorflow as tf
import os
import cv2
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

class RotResnet():
	def __init__(self, w=224, h=224, nc=4):
		self.img_width=w
		self.img_height=h
		self.num_classes=nc

	def plot_history(self, history, yrange):
		'''Plot loss and accuracy as a function of the epoch,
		for the training and validation datasets. '''
		acc = history.history['acc']
		val_acc = history.history['val_acc']
		loss = history.history['loss']
		val_loss = history.history['val_loss']

		# Get number of epochs
		epochs = range(len(acc))

		# Plot training and validation accuracy per epoch
		plt.plot(epochs, acc)
		plt.plot(epochs, val_acc)
		plt.title('Training and validation accuracy')
		plt.ylim(yrange)

		plt.savefig('Acc.png')

		# Plot training and validation loss per epoch
		plt.figure()

		plt.plot(epochs, loss)
		plt.plot(epochs, val_loss)
		plt.title('Training and validation loss')
		
		plt.savefig('Loss.png')

		# plt.show()

	def createRotResnet(self):
		vgg16 = keras.applications.vgg16
		conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

		x = keras.layers.Flatten()(conv_model.output)
		x = keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.01), bias_regularizer=keras.regularizers.l2(0.01))(x)
		x = keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.01), bias_regularizer=keras.regularizers.l2(0.01))(x)
		x = keras.layers.Dropout(0.35)(x)
		predictions = keras.layers.Dense(self.num_classes, activation='softmax')(x)

		full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)

		for layer in conv_model.layers:
			layer.trainable = False

		return full_model

	def train(self, model):
		sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
		data_generator=ImageDataGenerator(preprocessing_function=preprocess_input)
		test_datagen=ImageDataGenerator()

		train_generator=data_generator.flow_from_directory(
									directory='./split_data/train',
									classes=['-1','0','1','2'],
									# labels='inferred',
									# label_mode='int',
									color_mode='rgb',
									batch_size=64,
									seed=42,
									shuffle=True,
									interpolation='nearest',
									target_size=(self.img_height,self.img_width))

		# print(train_generator.samples)
		# print('Training set found.')

		valid_generator=data_generator.flow_from_directory(
									directory='./split_data/val',
									classes=['-1','0','1','2'],
									# labels='inferred',
									# label_mode='int',
									color_mode='rgb',
									batch_size=64,
									seed=42,
									shuffle=True,
									interpolation='nearest',
									target_size=(self.img_height,self.img_width))


		# print('Validation set found.')

		test_generator=test_datagen.flow_from_directory(
									directory='./split_data/test',
									classes=['-1','0','1','2'],
									# labels='inferred',
									# label_mode='int',
									color_mode='rgb',
									batch_size=1,
									seed=42,
									shuffle=False,
									interpolation='nearest',
									target_size=(self.img_height,self.img_width))

		# print('Test set found.')
		print('Training.')
		print()
		# print() 

		# x,y = train_generator.next()
		# for i in range(0,1):
		# 	image = x[i]
		# 	plt.imshow(image.transpose(2,1,0))
		# 	plt.show()
		model.summary()

		# opt = keras.optimizers.Adam(learning_rate=0.01)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

		# earlyStopping = EarlyStopping(monitor='val_loss', patience=, verbose=0, mode='min')
		mcp_save = ModelCheckpoint(os.path.join(os.getcwd(),'weights/mdl_wtsn8.hdf5'), save_best_only=True, monitor='val_loss', mode='min')
		reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4, mode='min')

		history = model.fit_generator(
			train_generator,
			validation_data = valid_generator,
			validation_steps=valid_generator.samples//valid_generator.batch_size,
			steps_per_epoch=train_generator.samples//train_generator.batch_size,
			callbacks=[mcp_save, reduce_lr_loss],
			epochs=10
		)

		print('Predicting')

		loss, acc = model.evaluate(test_generator, verbose=1, steps=test_generator.samples//test_generator.size)
		# print('On test set: ')
		print('Loss: ',loss)
		print('Accuracy: ',acc)

		# test_generator.reset()
		# predict = model.predict_generator(test_generator,steps = len(test_generator), verbose = 1)

		# predicted_class_indices=np.argmax(predict,axis=1)
		# labels = (train_generator.class_indices)
		# labels = dict((v,k) for k,v in labels.items())
		# predictions = [labels[k] for k in predicted_class_indices]

		# # print(predictions)
		# filenames=test_generator.filenames
		# results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
		# results.to_csv("results.csv",index=False)
		return history

if __name__ == '__main__':
	RRN=RotResnet()
	print()
	# print('Data prepped')
	model=RRN.createRotResnet()
	# print('Model created.')
	history=RRN.train(model)

	print('Plotting accuracy and loss graphs')
	RRN.plot_history(history, yrange=(0,1))

	print('Finished')

	# path="/home/mansi/Dossier_1_Lac_Data/RetinaFace/split_data/train/0/01012018_323451-Front.jpg"
	# img=cv2.imread(path, cv2.IMREAD_COLOR)

	# xs = np.expand_dims(img, axis=0)
	# print(xs.shape)
	# xs = keras.applications.vgg16.preprocess_input(xs)
	# features = model.predict(xs)
	# print(features[0])


