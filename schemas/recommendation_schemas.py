from pydantic import BaseModel, Field

class VisitedPlace(BaseModel):
    name: str
    country: str | None = None
    description: str | None = None
    
    
class RecommendPlacesRequest(BaseModel):
    visitedPlaces: list[VisitedPlace] = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=20)


class RecommendedPlace(BaseModel):
    name: str
    country: str
    description: str
    score: float
    imageUrl: str | None = None
    imageAuthor: str | None = None
    imageAuthorUrl: str | None = None
    imageSource: str | None = None


class RecommendPlacesResponse(BaseModel):
    recommendations: list[RecommendedPlace]