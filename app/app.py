from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', method=['GET'])
def get_articles():
    return jsonify({"Hello": "World"})

if __name__ == '__main__':
    app.run(debug=True)