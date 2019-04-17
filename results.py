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
	data = get_all_teams(5, True)
	# Hard coding this because the only user is me
	filepath = os.listdir(".")
	os.chdir("./5 games bac/with opponent/seperate")
	false_positives = []
	false_negatives = []
	true_positives = []
	true_negatives = []
	for i in range(1, 33):
		model = keras.models.load_model(str(i) + "-model.h5")
		fp = 0
		fn = 0
		tp = 0
		tn = 0
		for game in data:
			answer = game[1] == 1
			input_data = game[2:].reshape(1, 14)
			guess = model.predict(input_data) >= .5
			if answer:
				if guess:
					tp += 1
				else:
					fn += 1
			else:
				if guess:
					fp += 1
				else:
					tn += 1
		print("done " + str(i))
		false_negatives.append(fn)
		true_negatives.append(tn)
		false_positives.append(fp)
		true_positives.append(tp)
	print(true_positives)
	print(false_positives)
	print(true_negatives)
	print(false_negatives)

	plt.bar(np.arange(32.0)-.5, true_positives, width=0.25)
	plt.bar(np.arange(32.0)-.25, false_positives, width=0.25)
	plt.bar(np.arange(32.0), true_negatives, width=0.25)
	plt.bar(np.arange(32.0)+.25, false_negatives, width=0.25)
	plt.xticks(np.arange(32)-.125, np.arange(32))
	plt.ylabel('Count')
	plt.title('Confusion Matrix Counts')
	plt.legend(['True Positive', 'False Positive', 'True Negative', 'False Negative'])

	plt.show()

