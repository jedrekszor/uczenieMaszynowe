from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

k = 10
epochs = 1000
learning_rate = 0.001
reg = 0.01
TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/task.csv'
DEST_FILE = 'submission.csv'


def rmse(prediction, y):
    prediction = prediction[np.asarray(y != -1).nonzero()].flatten()
    y = y[np.asarray(y != -1).nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, y))


train = pd.read_csv(TRAIN_FILE, sep=";", names=["id", "user", "movie_id", "rating"])
test = pd.read_csv(TEST_FILE, sep=";", names=["id", "user", "movie_id", "rating"])

# normalize input
train["rating"] /= 5
ratings = train.pivot(index="user", columns="movie_id", values="rating")
n_users, n_movies = ratings.shape

# create svd matrices and populate them
P = np.random.uniform(0, 1, [k, n_users])
Q = np.random.uniform(0, 1, [k, n_movies])

scores = ratings.fillna(-1).to_numpy()
errors = []
users, movies = np.asarray(scores != -1).nonzero()

for epoch in range(epochs):
    print("Epoch nr ", epoch, " of ", epochs)
    for user, movie in zip(users, movies):
        error = scores[user, movie] - np.dot(P[:, user].T, Q[:, movie])
        P[:, user] += learning_rate * (error * Q[:, movie] - reg * P[:, user])
        Q[:, movie] += learning_rate * (error * P[:, user] - reg * Q[:, movie])
    errors.append(rmse(np.dot(P.T, Q), scores))

outputs = np.dot(P.T, Q)
result = pd.DataFrame.from_records(outputs, columns=ratings.columns)

for index, row in test.iterrows():
    print("Movie nr ", index, " of ", test.index)
    user = ratings.index.get_loc(row["user"])
    movie = row["movie_id"]
    score = result.loc[user, movie] * 5
    if score < 0:
        score = 0
    elif score > 5:
        score = 5
    test.loc[index, "rating"] = str(int(round(score)))
test.to_csv(DEST_FILE, sep=";", index=False, header=False)
