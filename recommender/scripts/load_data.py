import sqlite3
import polars as pl

movies_path = "./data/ml-latest-small/movies.csv"

df_movies = pl.read_csv(movies_path)

uri = "sqlite:///collab.db"

df_movies.write_database(table_name="movies", connection=uri)

