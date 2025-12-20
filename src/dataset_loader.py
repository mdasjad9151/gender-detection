"""
Component 2: Dataset Loading
Handles loading and organizing audio datasets
"""
import numpy as np
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    
    def __init__(self, feature_extractor):
 
        self.feature_extractor = feature_extractor
    
    def load_from_directory(self, data_dir: Path, 
                           classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:

        X = []
        y = []
        
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Loading dataset from {data_dir}")
        
        for label_idx, label_name in enumerate(classes):
            folder = data_dir / label_name
            
            if not folder.exists():
                logger.warning(f"Class folder not found: {folder}")
                continue
            
            # Find all audio files (wav and WAV)
            wav_paths = list(folder.rglob("*.wav")) + list(folder.rglob("*.WAV"))
            
            logger.info(f"Found {len(wav_paths)} files in {label_name}")
            
            # Extract features with progress bar
            for wav_file in tqdm(wav_paths, desc=f"Processing {label_name}"):
                try:
                    features = self.feature_extractor.extract_features(str(wav_file))
                    X.append(features)
                    y.append(label_idx)
                except Exception as e:
                    logger.warning(f"Skipping {wav_file}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Loaded dataset: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def load_from_file_list(self, file_paths: List[str], 
                           labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
     
        X = []
        y = []
        
        for path, label in zip(file_paths, labels):
            try:
                features = self.feature_extractor.extract_features(path)
                X.append(features)
                y.append(label)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
        
        return np.array(X), np.array(y)