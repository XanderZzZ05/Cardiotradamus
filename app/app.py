from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, redirect, url_for, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Cargar el modelo
modelo_path = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_knn.pkl')
modelo = joblib.load(modelo_path)
# Cargar el scaler
scaler_path = os.path.join(os.path.dirname(__file__), 'modelo', 'scaler.pkl')
scaler = joblib.load(scaler_path)

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
    data = request.json # Obtener datos JSON del formulario
    try:
        # Validar datos necesarios esten en Json
        required_fields = ['edad', 'sexo', 'tipo_dolor', 'presion_arterial',
                           'colesterol', 'azucar_sangre', 'resultados_ecg',
                           'frecuencia_cardiaca_maxima', 'angina_inducida',
                           'depresion_st', 'pendiente_st_ejercicio',
                           'vasos_principales_fluroscopia', 'talasemia']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Extraer y convertir los datos al formato que espera el modelo
        input_features = [
            int(data['edad']),
            1 if data['sexo'].lower() == 'masculino' else 0,
            int(data['tipo_dolor']),
            int(data['presion_arterial']),
            int(data['colesterol']),
            int(data['azucar_sangre']),
            int(data['resultados_ecg']),
            int(data['frecuencia_cardiaca_maxima']),
            1 if data['angina_inducida'].lower() == 'sí' else 0,
            float(data['depresion_st']),
            int(data['pendiente_st_ejercicio']),
            int(data['vasos_principales_fluroscopia']),
            int(data['talasemia'])
        ]

        # Convertir a array de numpy y estandarizar los datos de entrada usando el scaler ya entrenado
        input_features = np.array(input_features).reshape(1, -1)
        input_features = scaler.transform(input_features)

        # Realizar la predicción
        prediccion = modelo.predict(input_features)[0]  # Obtener el primer elemento de la predicción

        # Listas de consejos
        consejos_con_problemas = [
            "1. Mantén una dieta baja en grasas saturadas y azúcares.",
            "2. Realiza ejercicio regularmente, al menos 30 minutos al día.",
            "3. Controla tu presión arterial y colesterol con chequeos regulares.",
            "4. Evita el consumo de tabaco y alcohol.",
            "5. Consulta con un médico para un plan de tratamiento adecuado."
        ]

        consejos_sin_problemas = [
            "1. Mantén una dieta equilibrada rica en frutas y verduras.",
            "2. Haz ejercicio regularmente para mantener tu corazón sano.",
            "3. Controla tu peso y evita el sobrepeso.",
            "4. Realiza chequeos médicos de rutina.",
            "5. Mantén una buena salud mental y maneja el estrés."
        ]

        # Interpretar el resultado y proporcionar consejos
        if prediccion == 1:
            resultado = "El paciente presenta problemas cardíacos."
            consejos = consejos_con_problemas
        elif prediccion == 0:
            resultado = "El paciente no posee problemas cardíacos."
            consejos = consejos_sin_problemas
        else:
            resultado = "Resultado inesperado."  # Mensaje para cualquier otro resultado (opcional)
            consejos = []

        # Devolver el resultado e incluir consejos en formato JSON
        return jsonify({'resultado': resultado, 'consejos': consejos})

    except ValueError as ve:
        return jsonify({'error': f'Value error: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Manejo de errores para página no encontrada
def pag_no_encontrada(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    # Registrar el manejador de errores y ejecutar la aplicación
    app.register_error_handler(404, pag_no_encontrada)
    app.run(debug=True, port=5000)
