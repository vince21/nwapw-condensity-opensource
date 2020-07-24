# Condensity

Condensity is a Python project by Toby U., Spencer C. and Vincent W. that summarizes text with a novel extraction algorithm.

## Installation
First, clone the repo and install nginx
#### Mac
```bash
brew install nginx
```
#### Linux
```bash
sudo apt-get install nginx
```
Next, set up virtualenv
```bash
pip3 install virtualenv
cd app/
virtualenv venv
source venv/bin/activate
```
Install pip packages and download nltk data
```bash
pip3 install -r dependencies/requirements.txt
python3 dependencies/nltkPackages.py
```
Finally, serve with gunicorn
```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```
## Resources and Citations
Cite Vader sentiment analyzer in webpage:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
