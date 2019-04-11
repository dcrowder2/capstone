import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
import sqlite3
import matplotlib.pyplot as plt


def get_games(team_id):
	database = sqlite3.connect("data.db3")
	cur = database.cursor()

	cur.execute("Select * from games where winning_id = ? or losing_id = ?", (team_id, team_id))
	data = cur.fetchall()
	array = np.array(data)
	# wins = np.array([x for x in array if x[1] == team_id])
	# losses = np.array([x for x in array if x[2] == team_id])
	database.close()
	return array


def get_data(team_id, games_back, include_opponent, games):
	if include_opponent:
		ret_array = [team_id]
		for i in range(games_back):
			if games[i][1] == team_id:
				ret_array.extend([1, games[i+1][3], games[i+1][4]])
			else:
				ret_array.extend([0, games[i+1][4], games[i+1][3]])
		return np.array(ret_array)
	else:
		ret_array = [team_id]
		for i in range(games_back):
			if games[i][1] == team_id:
				ret_array.extend([1, games[i+1][3]])
			else:
				ret_array.extend([0, games[i+1][4]])

		return np.array(ret_array)


def get_all_data(team_id, include_opponent, games_back):
	ret_array = []
	games = get_games(team_id)
	# because the network needs a consistent input, if there isn't enough games to fill it completely it is left out
	while games.shape[0] > games_back:
		ret_array.append(get_data(team_id, games_back, include_opponent, games))
		games = games[games_back:]
	return np.array(ret_array)


def get_all_teams(games_back, include_opponent):
	ret_array = []
	for i in range(1, 33):
		ret_array.extend(get_all_data(i, include_opponent, games_back))
	return np.array(ret_array)


def split(data):
	split_point = (data.shape[0] * 4) // 5
	return np.array(data[:split_point]), np.array(data[split_point:])


def train(data, network, v):

	labels = data[:, 1].reshape(-1, 1)

	return network.fit(data[:, 2:], labels, epochs=1000, batch_size=100, verbose=v)


def test(data, network):

	labels = data[:, 1].reshape(-1, 1)

	score = network.evaluate(data[:, 2:], labels, batch_size=100)
	return score


if __name__ == '__main__':
	data_all = get_all_teams(1, True)
	train_plots = []
	test_plots = []
	train_data, test_data = split(data_all)

	model = Sequential()

	model.add(Dense(30, activation='sigmoid', input_dim=2, kernel_initializer='uniform'))
	model.add(Dropout(0.35))

	model.add(Dense(30, activation='sigmoid', kernel_initializer='uniform'))
	model.add(Dropout(0.35))

	model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

	model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	history = train(train_data, model, 1)

	accuracy = test(test_data, model)
	print(accuracy[1])
	model.save("all-model.h5")
	train_plots = np.array(train_plots)
	plt.plot(history.history["acc"])
	plt.title('Training Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')

	plt.show()
