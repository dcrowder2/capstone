import tensorflow
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout



def train(data, network, pixels, v):

	labels = pixels[:, 0].reshape(-1, 1)

	return network.fit(data, labels, epochs=1000, batch_size=250, verbose=v)


def test(data, network, pixels):

	labels = pixels[:, 0].reshape(-1, 1)

	score = network.evaluate(data, labels, batch_size=100)
	return score


def update_model(image, train_pix, network):

	return train(image, network, train_pix, 0)


if __name__ == '__main__':

		model = Sequential()


		model.add(Dense(10, activation='tanh', input_dim=12, kernel_initializer='uniform'))

		model.add(Dropout(0.35))


		model.add(Dense(8, activation='tanh', kernel_initializer='uniform'))
		model.add(Dropout(0.35))


		model.add(Dense(1, activation='tanh', kernel_initializer='uniform'))


		model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

		history = train(data, model, train_pixels, 1)

		accuracy = test(data, model, test_pixels)
		print(accuracy)
