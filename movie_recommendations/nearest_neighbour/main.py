import operator

from prepare_movie_data import get_movie_data
from scipy import spatial
from dateutil.parser import parse
import pandas as pd

TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/task.csv'
DEST_FILE = 'submission.csv'
K = 9


def binarize(data, unique):
    result = []

    for u in unique:
        if u in data:
            result.append(1)
        else:
            result.append(0)
    return result


def prepare_movie_data(movies):
    movies["genres"] = movies["genres"].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    movies["genres"] = movies["genres"].str.split(',')

    movies["cast"] = movies["cast"].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    movies["cast"] = movies["cast"].str.split(',')

    movies["keywords"] = movies["keywords"].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    movies["keywords"] = movies["keywords"].str.split(',')

    movies["production_countries"] = movies["production_countries"].str.strip('[]').str.replace(' ', '').str.replace(
        "'", '')
    movies["production_countries"] = movies["production_countries"].str.split(',')

    movies["director"] = movies["director"].str.strip('[]').str.replace(' ', '').str.replace("'", '')

    genre_list, cast_list, director_list, keyword_list, country_list, language_list = [], [], [], [], [], []

    for index, row in movies.iterrows():

        for genre in row["genres"]:
            if genre not in genre_list:
                genre_list.append(genre)

        for cast in row["cast"]:
            if cast not in cast_list:
                cast_list.append(cast)

        for keyword in row["keywords"]:
            if keyword not in keyword_list:
                keyword_list.append(keyword)

        for country in row["production_countries"]:
            if country not in country_list:
                country_list.append(country)

        if row["director"] not in director_list:
            director_list.append(row["director"])

        if row["original_language"] not in language_list:
            language_list.append(row["original_language"])

    movies["genres_binary"] = movies["genres"].apply(lambda x: binarize(x, genre_list))
    movies["cast_binary"] = movies["cast"].apply(lambda x: binarize(x, cast_list))
    movies["keywords_binary"] = movies["keywords"].apply(lambda x: binarize(x, keyword_list))
    movies["countries_binary"] = movies["production_countries"].apply(lambda x: binarize(x, country_list))
    movies["director_binary"] = movies["director"].apply(lambda x: binarize(x, director_list))
    movies["language_binary"] = movies["original_language"].apply(lambda x: binarize(x, language_list))

    return movies


def find_neighbours(movies, movie, user_movies):
    similarity = []
    for index, row in user_movies.iterrows():
        sim = calculate_similarity(movies, movie["movie_id"], row["movie_id"])
        similarity.append((row["movie_id"], sim, row["rating"]))

    similarity.sort(key=operator.itemgetter(1))

    return similarity[:K]


def calculate_similarity(movies, movie_id_a, movie_id_b):
    movie_a = movies.iloc[int(movie_id_a) - 1]
    movie_b = movies.iloc[int(movie_id_b) - 1]

    genres_similarity = spatial.distance.cosine(movie_a["genres_binary"], movie_b["genres_binary"])
    cast_similarity = spatial.distance.cosine(movie_a["cast_binary"], movie_b["cast_binary"])
    keywords_similarity = spatial.distance.cosine(movie_a["keywords_binary"], movie_b["keywords_binary"])
    countries_similarity = spatial.distance.cosine(movie_a["countries_binary"], movie_b["countries_binary"])
    director_similarity = spatial.distance.cosine(movie_a["director_binary"], movie_b["director_binary"])
    language_similarity = spatial.distance.cosine(movie_a["language_binary"], movie_b["language_binary"])

    budget_similarity = abs(movie_a["budget"] - movie_b["budget"])
    popularity_similarity = abs(movie_a["popularity"] - movie_b["popularity"])
    rating_similarity = abs(movie_a["vote_average"] - movie_b["vote_average"])

    release_similarity = abs(parse(movie_a["release_date"]) - parse(movie_b["release_date"]))

    return genres_similarity + cast_similarity + keywords_similarity + countries_similarity + director_similarity + language_similarity + budget_similarity + popularity_similarity + rating_similarity + release_similarity.days


movies = get_movie_data()
movies = prepare_movie_data(movies)

train = pd.read_csv(TRAIN_FILE, sep=";", names=["id", "user", "movie_id", "rating"])
test = pd.read_csv(TEST_FILE, sep=";", names=["id", "user", "movie_id", "rating"])

total = len(test.index)
for index, row in test.iterrows():
    print("Movie nr ", index, " of ", total)
    neighbours = pd.DataFrame(find_neighbours(movies, row, train.loc[train['user'] == row["user"]]), columns=["movie_id", "similarity", "rating"])
    test.loc[index, "rating"] = str(int(round(neighbours["rating"].mean())))

test.to_csv(DEST_FILE, sep=";", index=False, header=False)
