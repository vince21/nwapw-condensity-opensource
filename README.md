# Condensity

Condensity is a Python project by Toby U., Spencer C., and Vincent W. that summarizes text with a novel extraction algorithm.

## Installation
First, clone the repo

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
See NOTICE.txt
D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
Erkan, Gunes, and Drafomir R Radev. LexRank: Graph-Based Lexical Centrality as Salience in Text Summarization, 2004, www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html.
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
