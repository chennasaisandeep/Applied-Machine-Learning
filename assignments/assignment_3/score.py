# score.py
import joblib
import numpy as np

def score(text: str, model, threshold: float = 0.5) -> tuple:
    """
    Score a text using a trained model and determine if it's spam based on a threshold.
    
    Args:
        text: The text to classify
        model: A trained sklearn model/pipeline
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        tuple: (prediction as bool, propensity as float)
    """
    if not text.strip():
        return False, 0.0
        
    input_message = [text]
    # Predict probability using the pipeline (which includes the vectorizer)
    propensity = float(model.predict_proba(input_message)[0][1])
    
    # If propensity is NaN, set it to 0.0 to avoid misleading output
    if np.isnan(propensity):
        propensity = 0.0
        
    # Explicitly convert prediction to boolean
    prediction = bool(propensity >= threshold)
    
    return prediction, propensity

def load_model(model_path):
    """
    Load a trained model from a file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        The loaded model
    """
    model = joblib.load(model_path)
    return model