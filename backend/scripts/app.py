from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from main import simulator
import os

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

@app.route('/api/start-backend', methods=['POST'])
def start_backend():
    try:
        # Recebe os parâmetros do frontend
        parameters = request.json
        print(f"Parâmetros recebidos no backend: {parameters}")

        resultado = simulator(parameters)

        return jsonify({"message": "Simulação concluída com sucesso", "resultado": resultado})
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        return jsonify({"error": str(e)}), 500

# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def serve(path):
#     if path != "" and os.path.exists(app.static_folder + '/' + path):
#         return send_from_directory(app.static_folder, path)
#     else:
#         return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("Iniciando o servidor Flask...")
    app.run(debug=True)

#--------------------------------------------------------------------------------