from flask import Flask, request, jsonify
from ml import moderate_comment
from ia import get_toxicity

app = Flask(__name__)

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    comment = data.get('comment', '') if data else ''
    result = moderate_comment(comment)
    return jsonify(result)

# @app.route('/predict_sentiment', methods=['POST'])
# def predict_sentiment():
#     data = request.json
#     comment = data.get('comment', '') if data else ''
#     result = get_toxicity(comment)
#     return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
