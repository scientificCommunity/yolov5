from flask import Flask, request, jsonify
import test1

app = Flask(__name__)


@app.route('/')
def home():
    resources_pos = test1.main_logic()
    return resources_pos


@app.route('/test')
def test():
    return 'test'


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']


if __name__ == '__main__':
    app.run(debug=True, port=8000)
