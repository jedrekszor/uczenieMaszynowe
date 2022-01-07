import pandas as pd

TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/task.csv'
DEST_FILE = 'submission.csv'
neighbour_users = 5

train = pd.read_csv(TRAIN_FILE, sep=";", names=["id", "user", "movie_id", "rating"])
test = pd.read_csv(TEST_FILE, sep=";", names=["id", "user", "movie_id", "rating"])

users = train.pivot(index="movie_id", columns="user", values="rating")
correlated_users = users.corr(method="pearson")


def calculate_rating(row, connected):
    ratings = []
    ratings_of_movie = train.loc[train["movie_id"] == row["movie_id"], ["user", "rating"]]
    for user, correlation in connected.items():
        value = ratings_of_movie.loc[(train["user"] == user), "rating"]
        if len(value) > 0:
            ratings.append(value.iloc[0])
            if len(ratings) >= neighbour_users:
                break
    if len(ratings) == 0:
        return 3
    results = ratings[:neighbour_users]
    return sum(results) / len(results)


total = len(test.index)
for index, row in test.iterrows():
    print("Movie nr ", index, " of ", total)
    connected = correlated_users[row["user"]].copy()
    connected = connected.drop(labels=row["user"]).dropna()
    connected.sort_values(ascending=False, inplace=True)
    rating = calculate_rating(row, connected)
    test.loc[index, "rating"] = str(int(round(rating)))
test.to_csv(DEST_FILE, sep=";", index=False, header=False)
