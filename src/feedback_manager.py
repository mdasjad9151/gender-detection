"""
Component 5: Feedback Management
Handles user feedback collection for continuous learning
"""
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages user feedback for model improvement"""
    
    def __init__(self, config):

        self.config = config
        self.feedback_dir = Path(config.feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each class
        for label in self.config.label_map.values():
            class_dir = self.feedback_dir / label.lower()
            class_dir.mkdir(exist_ok=True)
    
    def save_feedback(self, audio_path: str, predicted_label: int,
                     correct_label: int, user_id: Optional[str] = None,
                     confidence: Optional[float] = None,
                     request_id: Optional[str] = None):

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            user_suffix = f"_user{user_id}" if user_id else ""
            
            # Determine if prediction was correct
            status = "correct" if predicted_label == correct_label else "corrected"
            
            # Get correct label directory
            correct_label_name = self.config.label_map[correct_label].lower()
            target_dir = self.feedback_dir / correct_label_name
            
            # Copy audio file
            source = Path(audio_path)
            
            # If request_id is provided, use it as filename
            if request_id:
                target_name = f"{request_id}{source.suffix}"
            else:
                target_name = f"{timestamp}_{status}{user_suffix}{source.suffix}"
                
            target_path = target_dir / target_name
            
            # If source is the same as target (already in feedback), don't copy, just maybe update metadata
            if source.resolve() != target_path.resolve():
                 if not source.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                 shutil.copy2(audio_path, target_path)
            
            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "request_id": request_id,
                "predicted_label": predicted_label,
                "predicted_class": self.config.label_map[predicted_label],
                "correct_label": correct_label,
                "correct_class": self.config.label_map[correct_label],
                "status": status,
                "user_id": user_id,
                "confidence": confidence,
                "original_path": str(audio_path),
                "saved_path": str(target_path)
            }
            
            metadata_path = target_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Feedback saved: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            raise
    
    def update_feedback(self, request_id: str, new_correct_label: int, user_id: Optional[str] = None):
        """
        Move a feedback file to the correct label directory and update its metadata.
        """
        try:
            # Find the file in any of the class directories
            found_path = None
            current_metadata = {}
            
            for label_name in self.config.label_map.values():
                class_dir = self.feedback_dir / label_name.lower()
                # Search for file with request_id
                # pattern: request_id.* (extension can vary)
                for p in class_dir.glob(f"{request_id}.*"):
                    if p.suffix != ".json":
                        found_path = p
                        break
                if found_path:
                    break
            
            if not found_path:
                logger.warning(f"Feedback/Prediction with ID {request_id} not found.")
                return False

            # Load existing metadata
            metadata_path = found_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    current_metadata = json.load(f)
            
            predicted_label = current_metadata.get("predicted_label", -1) # specific to how we saved it
            confidence = current_metadata.get("confidence")
            original_path = current_metadata.get("original_path", str(found_path))

            # If the new label is different from the current directory, we need to move it
            # But wait, we just re-call save_feedback with the NEW correct label
            # The save_feedback logic handles putting it in "correct_label" dir.
            
            # However, we must ensure we delete the old one if it moved directories?
            # actually save_feedback copies. 
            
            new_path = self.save_feedback(
                audio_path=str(found_path),
                predicted_label=predicted_label,
                correct_label=new_correct_label,
                user_id=user_id,
                confidence=confidence,
                request_id=request_id
            )
            
            # If the new path is different from found_path, delete the old one
            if Path(new_path).resolve() != found_path.resolve():
                logger.info(f"Moving feedback from {found_path} to {new_path}")
                found_path.unlink()
                if metadata_path.exists():
                     metadata_path.unlink()
            
            return True

        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            raise
    
    def get_feedback_stats(self) -> Dict:

        stats = {
            "total": 0,
            "by_class": {},
            "correct_predictions": 0,
            "corrected_predictions": 0
        }
        
        for label_name in self.config.label_map.values():
            label_dir = self.feedback_dir / label_name.lower()
            
            if not label_dir.exists():
                continue
            
            # Count audio files
            audio_files = list(label_dir.glob("*.wav")) + list(label_dir.glob("*.WAV"))
            
            # Count correct vs corrected
            correct = len([f for f in audio_files if "_correct_" in f.name])
            corrected = len([f for f in audio_files if "_corrected_" in f.name])
            
            stats["by_class"][label_name] = {
                "total": len(audio_files),
                "correct": correct,
                "corrected": corrected
            }
            
            stats["total"] += len(audio_files)
            stats["correct_predictions"] += correct
            stats["corrected_predictions"] += corrected
        
        return stats
    
    def clear_feedback(self, class_name: Optional[str] = None):

        if class_name:
            class_dir = self.feedback_dir / class_name.lower()
            if class_dir.exists():
                shutil.rmtree(class_dir)
                class_dir.mkdir()
                logger.info(f"Cleared feedback for class: {class_name}")
        else:
            for label_name in self.config.label_map.values():
                class_dir = self.feedback_dir / label_name.lower()
                if class_dir.exists():
                    shutil.rmtree(class_dir)
                    class_dir.mkdir()
            logger.info("Cleared all feedback data")