from flask import Flask, request, jsonify, render_template
import os
import requests
import time
import json
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Azure Configuración
AZURE_COMPUTER_VISION_ENDPOINT = "https://santti7ago.cognitiveservices.azure.com/"
AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY = "2fa0c0b449b24e659ff0a9bf27ab2f67"

# Inicialización del cliente de visión
client = ComputerVisionClient(
    AZURE_COMPUTER_VISION_ENDPOINT,
    CognitiveServicesCredentials(AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY)
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_data = image.read()

    try:
        # Llamada a Azure Computer Vision para extraer el texto
        result = client.read_in_stream(image_data, raw=True)
        operation_location = result.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        result = client.get_read_result(operation_id)

        # Esperar hasta que esté listo
        while result.status.lower() in ["notstarted", "running"]:
            time.sleep(1)
            result = client.get_read_result(operation_id)

        # Extracción de texto
        extracted_text = ""
        if result.status == "succeeded":
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    extracted_text += line.text + "\n"
        
        # Guardar en archivo JSON
        text_json = {'text': extracted_text}
        with open('extracted_text.json', 'w') as f:
            f.write(json.dumps(text_json))

        return jsonify(text_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/identify_objects', methods=['POST'])
def identify_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_data = image.read()

    try:
        # Llamada a Azure Computer Vision para detectar objetos
        detected_objects = client.detect_objects_in_stream(image_data)
        objects_list = []
        for obj in detected_objects.objects:
            objects_list.append({
                'name': obj.object_property,
                'confidence': obj.confidence
            })

        # Guardar en archivo JSON
        objects_json = {'objects': objects_list}
        with open('identified_objects.json', 'w') as f:
            f.write(json.dumps(objects_json))

        return jsonify(objects_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
