from pydantic import BaseModel, Field


class DescriptionRequest(BaseModel):
    placeName: str = Field(min_length=2, max_length=100)
    placeLocation: str | None = Field(default=None, max_length=100)
    maxLength: int = Field(default=300, ge=50, le=1000)


class DescriptionResponse(BaseModel):
    placeName: str
    description: str = Field(min_length=20, max_length=1000)