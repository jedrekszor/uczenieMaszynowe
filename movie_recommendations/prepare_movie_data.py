import pandas as pd
import tmdbsimple as tmdb
import os

tmdb.API_KEY = "311625fff9620e5ee7e580bd9b20578f"
DATA_SOURCE_FILE = '../data/movie.csv'
DATA_DESTINATION_FILE = '../data/movie_data.csv'

movies = pd.DataFrame(
    columns=["genres", "director", "cast", "vote_average", "keywords", "production_countries", "budget", "popularity"])
map_name = lambda mov: mov["name"]


def get_movie(movie_id, index):
    movie_raw = tmdb.Movies(movie_id)
    movie_info = movie_raw.info()
    movie_credits = movie_raw.credits()
    movie_keywords = movie_raw.keywords()

    genres = list(map(map_name, movie_info["genres"]))
    director = list(map(map_name, filter(lambda mov: mov["job"] == "Director", movie_credits["crew"])))
    cast = list(map(map_name, movie_credits["cast"]))
    vote_average = movie_info["vote_average"]
    keywords = list(map(map_name, movie_keywords["keywords"]))
    production_countries = list(map(map_name, movie_info["production_countries"]))
    budget = movie_info["budget"]
    popularity = movie_info["popularity"]

    movies.loc[index] = [genres, director, cast, vote_average, keywords, production_countries, budget, popularity]


def get_movie_data():
    if os.path.isfile(DATA_DESTINATION_FILE):
        return pd.read_csv(DATA_DESTINATION_FILE, sep=";")

    print("Downloading data...")
    initial_data = pd.read_csv(
        DATA_SOURCE_FILE, sep=";", names=[
            "id", "tmdb_id", "title"])

    for index, data in initial_data.iterrows():
        print("Movie nr ", index)
        get_movie(data["tmdb_id"], index)

    print("Saving to file...")
    movies.to_csv(DATA_DESTINATION_FILE, index=False)
    return movies
