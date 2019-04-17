import tensorflow
import numpy as np
import keras
import sqlite3
import matplotlib.pyplot as plt
import os


def get_games(team_id):
	database = sqlite3.connect("data.db3")
	cur = database.cursor()

	cur.execute("Select * from games where winning_id = ? or losing_id = ?", (team_id, team_id))
	data = cur.fetchall()
	array = np.array(data)
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


if __name__ == "__main__":
	data = get_all_teams(16, True)
	# Hard coding this because the only user is me
	false_positives = 0
	false_negatives = 0
	true_positives = 0
	true_negatives = 0
	model = keras.models.load_model("D:\\capstone\\16 Games back\\with opponent\\all-model.h5")
	for game in data:
		answer = game[1] == 1
		input_data = game[2:].reshape(1, 47)
		guess = model.predict(input_data) >= .5
		if answer:
			if guess:
				true_positives += 1
			else:
				false_negatives += 1
		else:
			if guess:
				false_positives += 1
			else:
				true_negatives += 1
	print(true_positives)
	print(false_positives)
	print(true_negatives)
	print(false_negatives)

	plt.bar([0, 1, 2, 3], [true_positives, false_positives, true_negatives, false_negatives])
	plt.xticks([0, 1, 2, 3], ["True positive", "False positive", "True negative", "False negative"])
	plt.ylabel('Count')
	plt.title('Confusion Matrix Counts')

	plt.show()

