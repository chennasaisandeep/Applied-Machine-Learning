# app.py
from flask import Flask, request, jsonify, render_template
import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score import score, load_model

# Path to the saved pipeline model (includes the vectorizer)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model', 'best_model_LogisticRegression.pkl')


# Load model once at startup
model = load_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        # Ensure threshold is a float
        threshold = float(data.get('threshold', 0.5))
        prediction, propensity = score(text, model, threshold)
        return jsonify({
            'prediction': bool(prediction),
            'propensity': float(propensity)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)