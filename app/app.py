from flask import Flask, render_template, redirect, request, url_for, flash, jsonify
from article import Summarizer
import gunicorn
import shelve
from werkzeug.utils import secure_filename
import boto3

app = Flask(__name__)


@app.route('/')
def home():
    s3 = boto3.resource('s3')
    s3.Bucket('nwapw-ips').put_object(Key="home", Body=request.remote_addr)
    return render_template("index.html")



@app.route('/news')
def news():
    s3 = boto3.resource('s3')
    s3.Bucket('nwapw-ips').put_object(Key="news", Body=request.remote_addr)
    news_db = shelve.open('news')
    try:
        return render_template("news.html", articles=news_db['data'])
    except KeyError:
        return render_template('index.html', errormsg="Sorry, we're updating the newsfeed right now. Try again in a "
                                                      "minute!")


@app.route('/results', methods=['GET', 'POST'])
def results():
    s3 = boto3.resource('s3')
    s3.Bucket('nwapw-ips').put_object(Key="results", Body=request.remote_addr)
    if request.method == 'POST':
        text = request.form['text']
        percent = request.form['percent']

        # catches empty inputs
        if not text:
            return render_template('index.html', errormsg="Please enter text, a link, or upload a file")
        if not percent:
            return render_template('index.html', errormsg="Please enter a reduction percentage")

        # catches empty percent field
        if percent.isnumeric():
            percent = int(percent) / 100
        else:
            percent = None
        summary = Summarizer(text)
        summary_text = summary.condense(percent)
        metrics = summary.condense_metrics(summary_text)
        summary_sentences = [sentence.strip() for sentence in summary_text.split('\n') if sentence.strip() != '']

        warnings = []
        # this could be an "in" statement plus a list of domains
        # bloomberg is a special case; works fine sometimes
        if summary.domain == 'www.bloomberg.com':
            # error that bloomberg throws (sometimes)
            if 'Please make sure your browser supports JavaScript and cookies and that you are not blocking ' \
               'them from loading.' in summary_text:
                warnings.append(f'This domain ({summary.domain}) is known to occasionally function poorly with the '
                                f'web scraper. Try pasting the text in from the website!')

        return render_template('results.html', summary_sentences=summary_sentences, metrics=metrics, warnings=warnings)
    else:
        return render_template('index.html')


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    # note that we set the 404 status explicitly
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
