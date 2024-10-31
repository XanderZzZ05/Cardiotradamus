from flask import Flask, render_template, redirect, url_for, request, jsonify
import joblib
import os
app=Flask(__name__)

modelo_path = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_knn.pkl')
modelo = joblib.load(modelo_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicciones', methods=['GET', 'POST'])
def predicciones():
    return render_template('prediccion.html')  # Página de predicciones

def pag_no_encontrada(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.register_error_handler(404, pag_no_encontrada)
    app.run(debug=True, port=5000)
    