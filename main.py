from flask import Flask, render_template, request, jsonify
import requests
from flask_toastr import Toastr

from pprint import pprint

pprint(globals())
pprint(locals())

app = Flask(__name__)
url_api = "http://127.0.0.1:5000/predict_sentiment"

@app.route('/')
def accueil():
    return render_template('accueil.html')

@app.route('/comment', methods=['GET'])
def comment_vue():
    return render_template('comment.html')

@app.route('/check_commentaire', methods=['POST'])
def check_commentaire():
    commentaire = request.form['comment']
    data = {'comment': commentaire}
    response = requests.post(url_api, json=data)

    if response.status_code == 200:
        
        reponse_api = response.json()
        if reponse_api == 0: resultat = "neg"
        elif reponse_api == 1: resultat = "pos"
        else : resultat = "neu"

    else:
        resultat = "Erreur lors de la requête à l'API."

    return render_template('comment.html', resultat_api=resultat)

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port="5001")
