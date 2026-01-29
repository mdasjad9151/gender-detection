from pydantic import BaseModel
from typing import Dict, Optional

class PredictionResponse(BaseModel):
    request_id: str
    prediction: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]
    audio_path: str

class FeedbackRequest(BaseModel):
    request_id: str
    correct_label: int
    user_id: Optional[str] = None
