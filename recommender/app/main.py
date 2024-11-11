from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import polars as pl

from typing import List

from recommender.models.data_models import Movie

app = FastAPI(app="Recommender API")

uri = "sqlite://collab.db"

df_movies = pl.read_database_uri("SELECT * FROM movies", uri=uri)


@app.get("/movies/{keyword}")
def search_movie_ids(keyword: str) -> List[Movie]:
    df_selection = df_movies.filter(
        pl.col("title").str.to_lowercase().str.contains(keyword)
    )

    df_selection = df_selection.rename({"movieId": "id"})

    return df_selection.to_dicts()

# TODO post user rating for movie id

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
