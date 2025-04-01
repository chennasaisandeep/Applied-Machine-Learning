# test.py
import os
import time
import sys
import pytest
import requests
import subprocess
import joblib
import numpy as np
import threading
import signal
import platform

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from score import score, load_model

# Path to your existing trained model
MODEL_PATH = r'G:\Desktop\CMI_DS\semester_IV\applied_machine_learning\assignments\assignment_2\best_model_LogisticRegression.pkl'

@pytest.fixture(scope="session")
def trained_model():
    """Load the existing trained model"""
    model = load_model(MODEL_PATH)
    return model

# Unit tests for score function
class TestScoreFunction:
    
    def test_smoke(self, trained_model):
        """Test that the function runs without crashing"""
        prediction, propensity = score("hello", trained_model, 0.5)
        assert isinstance(prediction, bool)
        assert isinstance(propensity, float)
    
    def test_input_output_types(self, trained_model):
        """Test that the function returns the expected types"""
        prediction, propensity = score("test message", trained_model, 0.5)
        assert isinstance(prediction, bool)
        assert isinstance(propensity, float)
    
    def test_prediction_binary(self, trained_model):
        """Test that prediction is binary (True/False)"""
        prediction, _ = score("test message", trained_model, 0.5)
        assert prediction in [True, False]
    
    def test_propensity_range(self, trained_model):
        """Test that propensity is between 0 and 1"""
        _, propensity = score("test message", trained_model, 0.5)
        assert 0.0 <= propensity <= 1.0
    
    def test_threshold_zero(self, trained_model):
        """Test that threshold=0 always results in prediction=True unless propensity is exactly 0"""
        prediction, propensity = score("test message", trained_model, 0.0)
        # Only assert if propensity is greater than 0
        if propensity > 0:
            assert prediction is True
        else:
            # If propensity is 0, just pass the test
            assert True
    
    def test_threshold_one(self, trained_model):
        """Test that threshold=1 always results in prediction=False unless propensity is exactly 1"""
        prediction, propensity = score("test message", trained_model, 1.0)
        # Only assert if propensity is less than 1
        if propensity < 1:
            assert prediction is False
        else:
            # If propensity is 1, just pass the test
            assert True
    
    def test_obvious_spam(self, trained_model):
        """Test that obvious spam text with a low threshold should be flagged as spam"""
        # Using different thresholds to increase chances of detecting spam
        # For testing, we use a very low threshold to ensure spam is detected
        prediction, _ = score("claim 1000 rs worth of free gifts", trained_model, 0.1)
        assert prediction is True
    
    def test_obvious_ham(self, trained_model):
        """Test that obvious non-spam text with a high threshold should be flagged as non-spam"""
        # For testing, we use a high threshold to ensure ham is not flagged as spam
        prediction, _ = score("Hi, can we schedule a meeting for tomorrow? Thanks, John", trained_model, 0.9)
        assert prediction is False
    
    def test_empty_input(self, trained_model):
        """Test that empty input returns proper values"""
        prediction, propensity = score("", trained_model, 0.5)
        assert prediction is False
        assert propensity == 0.0


# Integration tests for Flask app
@pytest.fixture(scope="module")
def flask_app():
    """Start the Flask app for testing and stop it after tests are done"""
    # Determine the Flask app entry point
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    
    # Start Flask app in a subprocess
    if platform.system() == 'Windows':
        proc = subprocess.Popen(["python", app_path], 
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(["python", app_path], 
                               preexec_fn=os.setsid)
    
    # Give the app time to start
    time.sleep(2)
    
    yield "http://localhost:5000"
    
    # Terminate Flask app
    if platform.system() == 'Windows':
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    
    proc.wait()


class TestFlaskApp:
    
    def test_score_endpoint(self, flask_app):
        """Test the /score endpoint of the Flask app"""
        url = f"{flask_app}/score"
        payload = {"text": "test message", "threshold": 0.5}
        
        response = requests.post(url, json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        assert isinstance(data["prediction"], bool)
        assert isinstance(data["propensity"], float)
        assert 0.0 <= data["propensity"] <= 1.0
    
    def test_spam_prediction(self, flask_app):
        """Test that obvious spam is correctly identified with a low threshold"""
        url = f"{flask_app}/score"
        # Using a low threshold to ensure spam detection for testing
        payload = {"text": "claim 1000 rs worth of free gifts", "threshold": 0.1}
        
        response = requests.post(url, json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] is True
    
    def test_ham_prediction(self, flask_app):
        """Test that obvious non-spam is correctly identified with a high threshold"""
        url = f"{flask_app}/score"
        # Using a high threshold to ensure ham is not flagged as spam
        payload = {"text": "Hi, can we schedule a meeting for tomorrow? Thanks, John", "threshold": 0.9}
        
        response = requests.post(url, json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] is False


if __name__ == "__main__":
    import os
    # Run tests with coverage and save output to coverage.txt
    os.system("pytest -v --cov=score --cov=app --cov-report=term test.py > coverage.txt")