# Importación de bibliotecas
from flask import Flask, request, jsonify, render_template
import numpy as np
from load import joblib
import os
from werkzeug.utils import secure_filename

# Cargar modelo
dt = joblib.load('./dt1.joblib')

# Creando app Flask
server = Flask(__name__)

# Definiendo rutas
@server.route('/predictjson', methods=['POST'])
def predictjson():
    # procesar los datos de entrada
    data = request.json
    
    print(data)
    
    inputData = np.array([data['pH'], data['sulphates'], data['alcohol']])
    
    resultado = dt.predict([inputData.reshape(1,-1)])
    
    return jsonify({"predidction": str(resultado[0])})

if __name__ == '__main__':
    server.run(debug=False, host='0.0.0.0', port=8080)