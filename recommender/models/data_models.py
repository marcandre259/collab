from pydantic import BaseModel


class Movie(BaseModel):
    id: int
    title: str
    genres: str