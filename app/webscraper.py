# NWAPW
# Spencer Chang, Toby Ueno, Vincent Wilson
# date: 8/04/20
# description: scrapes from news sites manually or automatically

from newspaper import Article
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError
import requests
from bs4 import BeautifulSoup
from datetime import datetime


def npr_scrape(url, write=False):
    """
    Takes a URL for an NPR article and returns the page text and info.
    :param url: NPR article link
    :type url: str
    :param write: If true, writes the article text to a file
    :type write: bool
    :return: Article title, list of authors, date published( datetime), text, and an image
    :rtype: dict
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # try/except makes sure page is of expected format: if not, goes to default scraper
    # find title
    try:
        title = soup.find('div', class_='storytitle').find('h1').text.strip()
    except AttributeError:
        return None

    # find author
    author = soup.find('p', class_='byline__name byline__name--block').text.strip()

    # find datetime
    date_text = soup.find('div', class_='dateblock').find('time')['datetime']
    date = datetime.strptime(date_text, '%Y-%m-%dT%X%z')

    # finds text and creates list of all images (only using one)
    text_elements = []
    images = []

    body = soup.find('div', id='storytext')
    for element in body.children:
        if element.name == 'p':
            text_elements.append(element)
        elif element.name == 'div':
            element_images = element.find_all('img')
            element_images = [img['data-original'] for img in element_images]
            images += element_images

    raw_text = '\n'.join([tag.text.strip() for tag in text_elements])

    output_dict = {'Title': title,
                   'Authors': [author],
                   'Date': date,
                   'Text': raw_text,
                   'Image': images[0]
                   }

    # writes text to file (useful in testing/debugging to avoid scraping each time)
    if write:
        modified_title = '-'.join(title.lower().split(' '))
        with open(f'scraped-text/{modified_title}.txt', 'w') as f:
            f.write(raw_text)

    return output_dict


def wapo_scrape(url):
    """
    Takes a URL for a Washington Post article and returns the page text and info.
    :param url: Washington Post article link
    :type url: str
    :return: Article title, list of authors, and text
    :rtype: dict
    """
    # this scraper (and all other manual ones) relegate date and image finding to default_scrape

    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # title and format check (same as npr_scrape)
    try:
        title = soup.find('h1', class_='font--headline').text.strip()
    except AttributeError:
        return None

    # find authors
    authors = soup.find_all('span', class_='author-name')
    authors = [author.text.strip() for author in authors]

    # find text
    body_text = soup.find_all('p', class_='font--body')
    raw_text = '\n'.join([tag.parent.text.strip() for tag in body_text])

    output_dict = {'Title': title,
                   'Authors': authors,
                   'Text': raw_text
                   }

    return output_dict


def bbc_scrape(url):
    """
    Takes a URL for a BBC article and returns the page text and info.
    :param url: BBC article link
    :type url: str
    :return: Article title, list of authors, and text
    :rtype: dict
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # find title and check format
    try:
        title = soup.find('h1', class_='story-body__h1').text.strip()
    except AttributeError:
        return None

    # find authors
    try:
        authors = [soup.find('span', class_='byline__name').text.strip()[3:]]
    except AttributeError:
        # if no authors are specified, will raise Attribute Error
        authors = []

    # find text
    body_text = soup.find('div', class_='story-body__inner').find_all('p', recursive=False)
    # removes ad manually (only appears on some articles)
    if body_text[-1].text[:21] == 'Follow us on Facebook':
        body_text = body_text[:-1]
    raw_text = '\n'.join([tag.text.strip() for tag in body_text])

    output_dict = {'Title': title,
                   'Authors': authors,
                   'Text': raw_text
                   }

    return output_dict


def atlantic_scrape(url):
    """
    Takes a URL for an Atlantic article and returns the page text and info.
    :param url: Atlantic article link
    :type url: str
    :return: Article title, list of authors, and text
    :rtype: dict
    """
    # doesn't work for all articles
    # ex: https://www.theatlantic.com/magazine/archive/2020/09/coronavirus-american-failure/614191/
    # however, default scraper seems to capture them okay
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # find title / format check
    try:
        title = soup.find('h1', class_='c-article-header__hed').text.strip()
    except AttributeError:
        return None

    # find authors
    authors = soup.find_all('span', class_='c-byline__author')
    authors = [author.text.strip() for author in authors]

    # find text
    body_text = soup.find_all('p', {'dir': 'ltr', 'data-id': False})
    raw_text = '\n'.join([tag.text.strip() for tag in body_text])

    output_dict = {'Title': title,
                   'Authors': authors,
                   'Text': raw_text}

    return output_dict


def is_valid_url(url):
    """
    Tests if a URL is valid by checking the syntax and the response code.
    :param url: String to be tested for URL validity.
    :type url: str
    :return: True if URL is valid, False otherwise
    :rtype: bool
    """
    valid_syntax = False
    try:
        parsed_url = urlparse(url)
        valid_syntax = all([parsed_url.scheme, parsed_url.netloc])
    except ValueError:
        # thrown on error with parsing url
        pass
    if valid_syntax:
        try:
            return urlopen(url).getcode() == 200
        except URLError:
            # occurs when url can't be opened
            pass
    return False


def default_scrape(url, data={}):
    """
    Fills in missing entries from an article info dict with info from newspaper3k. Non-destructive.
    :param url: Valid url for a web article (will return None if detected invalid)
    :type url: str
    :param data: Dictionary to be filled with Title, Author, Date, Text, and Image keys.
    :type data: dict
    :return: New dictionary with all article info keys (when possible).
    :rtype: dict
    """
    article = Article(url)
    article.download()
    article.parse()
    # add info from input dict
    article_info = {}
    for k, v in data.items():
        if k in ['Title', 'Authors', 'Date', 'Text', 'Image']:
            article_info[k] = v
    # add defaults from parsed Article if not already in article_info dict
    article_info['Title'] = article_info.get('Title', article.title)
    article_info['Authors'] = article_info.get('Authors', article.authors)
    article_info['Date'] = article_info.get('Date', article.publish_date)
    article_info['Text'] = article_info.get('Text', article.text)
    article_info['Image'] = article_info.get('Image', article.top_image)

    return article_info


def scrape(url):
    """
    Takes a url and returns info about the page.
    :param url: Valid url for a web article (will return None if detected invalid)
    :type url: str
    :return: Article title, list of authors, date published (datetime), text, and an image
    :rtype: dict
    """

    if not is_valid_url(url):
        return None

    # apply custom scrapers if possible
    domain = urlparse(url).netloc.split('.')[1]

    if domain == 'npr':
        output_dict = npr_scrape(url)
    elif domain == 'washingtonpost':
        output_dict = wapo_scrape(url)
    elif domain == 'bbc':
        output_dict = bbc_scrape(url)
    elif domain == 'theatlantic':
        output_dict = atlantic_scrape(url)

    if not output_dict:
        output_dict = {}

    # fill in gaps with newspaper module
    output_dict = default_scrape(url, output_dict)

    return output_dict


if __name__ == '__main__':
    test_url = 'https://www.theatlantic.com/magazine/archive/2020/09/coronavirus-american-failure/614191/'
    scrape_output = scrape(test_url)
    print(f'Title: {scrape_output["Title"]}')
    print(f'Authors: {scrape_output["Authors"]}')
    print(f'Date: {scrape_output["Date"]}')
    print(f'Text: {scrape_output["Text"]}')
    print(f'Image: {scrape_output["Image"]}')
