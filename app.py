from flask import Flask
from flask import render_template, request
import process

app = Flask("Wagony_endpoint")

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        responses = []
        for file in request.files.getlist('file'):
            if file.filename != '':
                responses.append(process.extract_flask(file))
        return render_template("index.html", resp=responses)
    return render_template("index.html")
