from typing import Optional
from src.config import ModelConfig
from src.facade import GenderDetectionFacade
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

def create_detector(config: Optional[ModelConfig] = None) -> GenderDetectionFacade:

    return GenderDetectionFacade(config)


if __name__ == "__main__":
    print("GenderDetectionFacade - Quick Test")
    print("-" * 60)
    
    # Initialize
    detector = GenderDetectionFacade()
    
    # Check if model exists
    if detector.is_model_trained():
        print(" Trained model found")
        info = detector.get_model_info()
        print(f"  Accuracy: {info['config'].get('metrics', {}).get('accuracy', 'N/A')}")
    else:
        print("No trained model found")
        print("Training initial model...")
        data_dir = Path("data")
        metrics = detector.train_initial_model(
            data_dir=str(data_dir),
            classes=["female", "male"]
        )
        print(f" Model Training Complete!")
        print()
        print(f"Performance Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print()