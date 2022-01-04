import pandas as pd
from prepare_movie_data import get_movie_data
from Classifier import Classifier
from anytree.dotexport import DotExporter

TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/task.csv'
DEST_FILE = 'submission.csv'

movies = get_movie_data()
movies = movies[["vote_average", "budget", "popularity"]]
train = pd.read_csv(TRAIN_FILE, sep=";", names=["id", "user", "movie_id", "rating"])
test = pd.read_csv(TEST_FILE, sep=";", names=["id", "user", "movie_id", "rating"])

users = test["user"].unique()
i, total = 0, len(users)
for user in users:
    i += 1
    print(i, "/", total)

    raw_data = train[train["user"] == user]
    user_movie_ids = raw_data.apply(lambda data: data["movie_id"], axis=1)
    user_movie_data = user_movie_ids.apply(lambda d: movies.iloc[d - 1])

    train_x = user_movie_data
    train_y = pd.DataFrame(raw_data["rating"], columns=["rating"])

    tree = Classifier(split_stop=2, depth_stop=3)
    tree.fit(train_x, train_y)

    raw_test_data = test[test["user"] == user]
    user_test_movie_ids = raw_test_data.apply(lambda data: data["movie_id"], axis=1)
    user_test_movie_data = user_test_movie_ids.apply(lambda d: movies.iloc[int(d) - 1])

    test_x = user_movie_data
    test_y = tree.predict(test_x)

    indexes = raw_test_data["id"]
    j = 0
    for index in indexes:
        test.loc[test["id"] == index, "rating"] = str(test_y[j])
        j += 1

    if i == 1:
        tree_png = tree.generate_tree_png()
        DotExporter(tree_png).to_picture("tree.png")
        print("Tree png saved to file")
        exit(0)

test.to_csv(DEST_FILE, sep=";", index=False, header=False)
