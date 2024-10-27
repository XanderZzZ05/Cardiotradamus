from flask import Flask, render_template, redirect, url_for, request, jsonify
import joblib

app=Flask(__name__)

modelo=joblib.load('C:/Users/Vivia/OneDrive/Escritorio/CARDIOTRADAMUS/app/modelo/modelo_knn.pkl')

@app.route('/')
def index():
    cursos=[1,2,3,4,5,6]
    data={
        'titulo':'index123',
        'bienvenida':'saludos',
        'cursos': cursos,
        'numero_cursos':len(cursos)
    }
    return render_template('index.html', data=data)

@app.route('/predicciones', methods=['GET', 'POST'])
def predict():
    return render_template('prediccion.html')  # Página de predicciones

"""@app.route('/api/prediccion', methods=['POST'])
def realizar_prediccion():
    data = request.get_json()
    input1 = data.get('input1')
    input2 = data.get('input2')
    
    # Aquí iría la lógica para realizar la predicción
    resultado = "Resultado de la predicción"  # Reemplaza esto con tu lógica

    return jsonify({'resultado': resultado})
"""
def pag_no_encontrada(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.register_error_handler(404, pag_no_encontrada)
    app.run(debug=True, port=5000)
    