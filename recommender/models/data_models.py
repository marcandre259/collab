from pydantic import BaseModel


class Movie(BaseModel):
    movie_id: int
    title: str
    genres: str


class Review(BaseModel):
    user: str
    movie_id: int
    review_score: float
