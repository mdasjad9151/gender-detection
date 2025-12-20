"""
Component 1: Audio Feature Extraction
Handles audio loading and MFCC feature extraction
"""
import librosa
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extracts MFCC features from audio files"""
    
    def __init__(self, config):
   
        self.config = config
        self.sample_rate = config.sample_rate
        self.duration = config.duration
        self.n_mfcc = config.n_mfcc
    
    def load_audio_fixed_length(self, path: str) -> Tuple[np.ndarray, int]:

        try:
            # Load audio
            y, sr = librosa.load(path, sr=self.sample_rate)
            
            # Calculate target length
            target_len = int(self.sample_rate * self.duration)
            
            # Truncate or pad
            if len(y) > target_len:
                y = y[:target_len]
            elif len(y) < target_len:
                pad_width = target_len - len(y)
                y = np.pad(y, (0, pad_width), mode='constant')
            
            return y, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio from {path}: {e}")
            raise
    
    def extract_features(self, audio_path: str) -> np.ndarray:

        try:
            # Load audio
            y, sr = self.load_audio_fixed_length(audio_path)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc
            )
            
            # Calculate statistics
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            
            # Concatenate features
            features = np.concatenate([mfcc_mean, mfcc_std], axis=0)
            
            logger.debug(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            raise
    
    def extract_features_batch(self, audio_paths: list) -> np.ndarray:

        features_list = []
        
        for path in audio_paths:
            try:
                features = self.extract_features(path)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
        
        return np.array(features_list)