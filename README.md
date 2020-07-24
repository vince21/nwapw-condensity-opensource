# article-summarizer

To run on your computer:
Go to the app folder
pip3 install virtualenv
source venv/bin/activate
pip3 install -r dependencies/requirements.txt
python3 dependencies/nltkPackages.py
gunicorn --bind 0.0.0.0:5000 wsgi:app
deactivate

Cite Vader sentiment analyzer in webpage:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
