from flask import Flask, render_template, redirect, url_for, request, jsonify
import joblib
import os

app = Flask(__name__)

# Cargar el modelo
modelo_path = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_knn.pkl')
modelo = joblib.load(modelo_path)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la página de predicciones
@app.route('/predicciones', methods=['GET'])
def predicciones():
    return render_template('prediccion.html')  # Página HTML para ingresar datos de predicción

# API para hacer la predicción
@app.route('/api/prediccion', methods=['POST'])
def api_prediccion():
    data = request.json  # Obtener datos JSON del formulario
    try:
        # Extraer y convertir los datos al formato que espera el modelo
        input_features = [
            float(data['edad']),
            1 if data['sexo'].lower() == 'masculino' else 0,
            float(data['tipo_dolor']),
            float(data['presion_arterial']),
            float(data['colesterol']),
            float(data['azucar_sangre']),
            float(data['resultados_ecg']),
            float(data['frecuencia_cardiaca_maxima']),
            1 if data['angina_inducida'].lower() == 'sí' else 0,
            float(data['depresion_st']),
            float(data['pendiente_st_ejercicio']),
            int(data['vasos_principales_fluroscopia']),
            1 if data['talasemia'].lower() == 'sí' else 0
        ]
        
        # Realizar la predicción
        prediccion = modelo.predict([input_features])[0]  # Obtener el primer elemento de la predicción
        
        # Interpretar el resultado
        if prediccion == 1:
            resultado = "El paciente presenta un riesgo de problemas cardíacos."
        else:
            resultado = "El paciente no presenta riesgo de problemas cardíacos."
        
        # Devolver el resultado interpretado en formato JSON
        return jsonify({'resultado': resultado})

    except Exception as e:
        # En caso de error, devolver el mensaje de error
        return jsonify({'error': str(e)}), 400

# Manejo de errores para página no encontrada
def pag_no_encontrada(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    # Registrar el manejador de errores y ejecutar la aplicación
    app.register_error_handler(404, pag_no_encontrada)
    app.run(debug=True, port=5000)
