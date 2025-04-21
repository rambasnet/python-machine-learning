from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('first_app.html')


@app.route('/hello/<name>')
def hello(name):
    return f'<p>Hello, {name}!</p>'


if __name__ == '__main__':
    app.run(debug=True)
