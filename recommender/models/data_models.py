from pydantic import BaseModel


class Movie(BaseModel):
    movie_id: int
    title: str
    genres: str


class Review(BaseModel):
    user: str
    movie_id: int
    review_score: float


class MovieReviewResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    user: str
    review_score: float

    class Config:
        schema_extra = {
            "example": {
                "movie_id": 1,
                "title": "The Shawshank Redemption",
                "genres": "Drama",
                "user": "john_doe",
                "review_score": 4.5
            }
        }