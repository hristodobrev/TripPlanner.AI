from pydantic import BaseModel, Field


class RecommendPlacesRequest(BaseModel):
    visitedPlaces: list[str] = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=20)


class RecommendedPlace(BaseModel):
    name: str
    country: str
    description: str
    score: float


class RecommendPlacesResponse(BaseModel):
    recommendations: list[RecommendedPlace]