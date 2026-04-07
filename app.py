# app.py

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict_match
from train import retrain_after_match

DATA_PATH = 'data/matches.csv'

app = FastAPI(title="Soccer Prediction API")


class MatchRequest(BaseModel):
    team_A: str
    team_B: str


class MatchResult(BaseModel):
    team_A:  str
    team_B:  str
    goals_A: int
    goals_B: int
    date:    str


@app.post("/predict")
def predict(request: MatchRequest):
    try:
        df     = pd.read_csv(DATA_PATH, parse_dates=['date'])
        result = predict_match(request.team_A, request.team_B, df)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Match data not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/result")
def submit_result(result: MatchResult):
    """
    Call this after a real match is played.
    Appends result to CSV and retrains model.
    """
    try:
        retrain_after_match(
            team_A  = result.team_A,
            team_B  = result.team_B,
            goals_A = result.goals_A,
            goals_B = result.goals_B,
            date    = result.date
        )
        return {"status": "ok", "message": "Match saved and model retrained."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}