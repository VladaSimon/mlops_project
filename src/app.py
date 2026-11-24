import pickle
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

FEATURES = ["latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights", "neighbourhood_cleansed", "property_type", "room_type"]

MODEL_PATH = Path("/artifacts/model.pkl")

def frame(rows):
    return pd.DataFrame(rows, columns=FEATURES)

@asynccontextmanager
async def lifespan(app):
    with MODEL_PATH.open("rb") as file:
        app.state.model = pickle.load(file)
    yield

app = FastAPI(title="Airbnb API", lifespan=lifespan)

class AirbnbInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    latitude: FiniteFloat
    longitude: FiniteFloat
    accommodates: FiniteFloat = Field(gt=0)
    bathrooms: FiniteFloat = Field(ge=0)
    bedrooms: FiniteFloat = Field(ge=0)
    beds: FiniteFloat = Field(ge=0)
    minimum_nights: FiniteFloat = Field(gt=0)
    neighbourhood_cleansed: str
    property_type: str
    room_type: str

@app.post("/predict")
def predict(data: AirbnbInput):
    result = app.state.model.predict(frame([data.model_dump()]))
    return {"prediction": float(result[0])}