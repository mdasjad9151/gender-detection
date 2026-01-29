from fastapi import FastAPI, UploadFile, File, HTTPException
from src.facade import GenderDetectionFacade
from backend.schemas import PredictionResponse, FeedbackRequest
import uuid6
import shutil
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gender Detection API")

# Initialize Facade
# We generally want to load the model once at startup
detector = GenderDetectionFacade()
if not detector.is_model_trained():
    logger.warning("No trained model found. Please train the model first.")
    # You might want to trigger training here or just warn
    # For now, we assume a model exists or will be trained via other means
    # But for the API to work, we need a model.
    # We can try to load it or fail gracefully.

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Gender Detection API")
    if detector.is_model_trained():
        detector.get_model_info() # Triggers loading config
    else:
        logger.warning("Model not trained.")

@app.get("/test")
def test_endpoint():
    return {"message": "Backend is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload an audio file for gender prediction.
    Automatically saves the result as 'presumed correct' in feedback data.
    """
    request_id = str(uuid6.uuid7())
    
    # Save temporary file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{request_id}{Path(file.filename).suffix}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Predict
        # We need to ensure the model is loaded. 
        # The facade loads it on first predict call if not loaded.
        try:
            result = detector.predict(str(file_path))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        # Save as feedback (presumed correct)
        # We assume the prediction is correct initially
        predicted_label_id = result["label_id"]
        
        # Save to feedback system
        saved_path = detector.feedback_manager.save_feedback(
            audio_path=str(file_path),
            predicted_label=predicted_label_id,
            correct_label=predicted_label_id, # Presumed correct
            confidence=result["confidence"],
            request_id=request_id
        )
        
        # Construct response
        return PredictionResponse(
            request_id=request_id,
            prediction=result["prediction"],
            label_id=result["label_id"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            audio_path=str(file_path) # Returning the temp path or saved path? 
                                      # Maybe clear to return request_id as primary ref
        )
        
    finally:
        # Clean up temp file? 
        # feedback_manager.save_feedback copies the file. 
        # So we can remove the temp file.
        if file_path.exists():
            file_path.unlink()

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Correct a prediction if it was wrong.
    """
    success = detector.feedback_manager.update_feedback(
        request_id=feedback.request_id,
        new_correct_label=feedback.correct_label,
        user_id=feedback.user_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Request ID not found")
        
    return {"message": "Feedback updated successfully"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_trained": detector.is_model_trained()}
