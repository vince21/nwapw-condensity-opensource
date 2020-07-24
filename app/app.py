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
        # catches empty percent field
        if percent.isnumeric():
            percent = int(percent) / 100
        else:
            percent = None
        summary = Summarizer(text)
        summary_text = summary.condense(percent)
        metrics = summary.condense_metrics(summary_text)
        return render_template('results.html', summary_text=summary_text, metrics=metrics)
    else:
        return render_template('index.html')


if __name__ == '__main__':
   app.run(host='0.0.0.0')
