from flask import Flask, request, jsonify, render_template

# todo: downgrade version sklearn to 1.0.2

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8114)
