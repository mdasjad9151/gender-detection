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
                     confidence: Optional[float] = None):

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
            if not source.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            target_name = f"{timestamp}_{status}{user_suffix}{source.suffix}"
            target_path = target_dir / target_name
            
            shutil.copy2(audio_path, target_path)
            
            # Save metadata
            metadata = {
                "timestamp": timestamp,
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
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
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