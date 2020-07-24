from flask import Flask, render_template, redirect, request
from article import Summarizer
import gunicorn

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        text = request.form['text']
        percent = request.form['percent']
        summary = Summarizer(text)
        return summary.condense(int(percent)/100)
    else:
        return render_template('index.html')


if __name__ == '__main__':
   app.run(host='0.0.0.0')
