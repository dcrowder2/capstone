# import tensorflow
# import keras
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
import sqlite3


def get_games(team_id):
	database = sqlite3.connect("data.db3")
	cur = database.cursor()

	cur.execute("Select * from games where winning_id = ? or losing_id = ?", (team_id, team_id))
	data = cur.fetchall()
	array = np.array(data)
	wins = np.array([x for x in array if x[1] == team_id])
	losses = np.array([x for x in array if x[2] == team_id])
	database.close()
	return array


def get_data(team_id, games_back, include_opponent):
	games = get_games(team_id)
	if include_opponent:
		ret_array = []
		for i in range(games_back):
			if games[i][1] == team_id:
				ret_array.append([1, games[i][3], games[i][4]])
			else:
				ret_array.append([0, games[i][4], games[i][3]])
		return np.array(ret_array)
	else:
		ret_array = []
		for i in range(games_back):
			if games[i][1] == team_id:
				ret_array.append([1, games[i][3]])
			else:
				ret_array.append([0, games[i][4]])
		return np.array(ret_array)


# def train(data, network, pixels, v):
#
# 	labels = pixels[:, 0].reshape(-1, 1)
#
# 	return network.fit(data, labels, epochs=1000, batch_size=250, verbose=v)
#
#
# def test(data, network, pixels):
#
# 	labels = pixels[:, 0].reshape(-1, 1)
#
# 	score = network.evaluate(data, labels, batch_size=100)
# 	return score
#
#
# def update_model(image, train_pix, network):
#
# 	return train(image, network, train_pix, 0)


if __name__ == '__main__':
	data1 = get_data(29, 200, True)
	data2 = get_data(29, 100, False)
	print("yeah")

	# model = Sequential()
	#
	#
	# model.add(Dense(10, activation='tanh', input_dim=12, kernel_initializer='uniform'))
	#
	# model.add(Dropout(0.35))
	#
	#
	# model.add(Dense(8, activation='tanh', kernel_initializer='uniform'))
	# model.add(Dropout(0.35))
	#
	#
	# model.add(Dense(1, activation='tanh', kernel_initializer='uniform'))
	#
	#
	# model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	#
	# history = train(data, model, train_pixels, 1)
	#
	# accuracy = test(data, model, test_pixels)
	# print(accuracy)
