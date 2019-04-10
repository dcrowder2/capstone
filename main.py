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
		ret_array.append(get_all_data(i, include_opponent, games_back))
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
	data_all = get_all_teams(5, False)
	train_plots = []
	test_plots = []
	for team in data_all:
		train_data, test_data = split(team)

		model = Sequential()

		model.add(Dense(10, activation='tanh', input_dim=9, kernel_initializer='uniform'))
		model.add(Dropout(0.35))

		model.add(Dense(8, activation='tanh', kernel_initializer='uniform'))
		model.add(Dropout(0.35))

		model.add(Dense(1, activation='tanh', kernel_initializer='uniform'))

		model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

		history = train(train_data, model, 1)

		accuracy = test(test_data, model)

		# plot_model(model)
		temp = [float(team[0][0])]
		temp.extend(history.history["acc"])
		train_plots.append(temp)
		test_plots.append(accuracy[1])
		print(accuracy[1])
		model.save(str(team[0][0]) + "-model.h5")
	train_plots = np.array(train_plots)
	plt.figure(1)
	for plot in train_plots:
		plt.plot(plot[1:])
	plt.legend(train_plots[:, 0], loc='upper left')
	plt.title('Training Accuracy by Team')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')

	plt.figure(2)
	plt.bar(np.arange(32), test_plots, align='center', alpha=0.5)
	plt.xticks(np.arange(32), train_plots[:, 0])
	plt.ylabel('Accuracy')
	plt.title('Validation Accuracy per Team')

	plt.show()
