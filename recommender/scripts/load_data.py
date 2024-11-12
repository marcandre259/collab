import sqlite3
import polars as pl

movies_path = "./data/ml-latest-small/movies.csv"

df_movies = pl.read_csv(movies_path)

uri = "sqlite:///collab.db"

try:
    df_movies.write_database(table_name="movies", connection=uri)
except ValueError:
    print("Table 'movies' already exists")

# Create reviews table
conn = sqlite3.connect("collab.db")

cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT NOT NULL,
        movie_id INTEGER NOT NULL,
        review_score FLOAT NOT NULL
    )
    """
)

conn.commit()
