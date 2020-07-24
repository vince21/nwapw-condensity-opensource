from newspaper import Article
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def npr_scrape(url):
    """
    Takes a URL for an NPR article and returns the text and images.
    :param url: NPR article link
    :type url: str
    :param write: If true, writes the article text to a file
    :type write: bool
    :return: List of text elements in body and list of image links
    :rtype: ([bs4.element.Tag], [str]))
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    title = soup.find('div', class_='storytitle').find('h1').text.strip()

    author = [element.text.strip() for element in soup.find_all('a', rel='author')]

    date_text = soup.find('div', class_='dateblock').find('time')['datetime']
    date = datetime.strptime(date_text, '%Y-%m-%dT%X%z')

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
                   'Author': author,
                   'Date': date,
                   'Text': raw_text,
                   'Image': images[0]
                   }

    return output_dict

def cnn_scrape(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    title = soup.find('h1', class_='pg-headline').text.strip()

    author = soup.find('span', class_='metadata__byline__author').text
    author_list = author.split(', ')
    author_list[0] = author_list[0][3:] # gets rid of "by"
    del author_list[-1] # deletes "cnn" at end
    if ' and ' in author:
        ending_authors = author_list[-1].split(' and ')
        del author_list[-1]
        for ending_author in ending_authors:
            author_list.append(ending_author)

    # strftime doesn't recognize ET as timezone; timezone not included
    date_text = soup.find('p', class_='update-time').text.strip().split('Updated ')[-1].split(', ')
    date_text[0] = date_text[0][:-3]
    date_text = ' '.join(date_text)
    print(date_text)
    date = datetime.strptime(date_text, '%M:%S %p %a %B %d %Y')
    print(date)

    body = soup.find_all(class_='zn-body__paragraph')
    body = [tag for tag in body if not tag.find('h3')]
    raw_text = '\n'.join([tag.text.strip() for tag in body])
    if raw_text[:5] == '(CNN)':
        raw_text = raw_text[5:]

    image = soup.find('img', class_='media__image')['src']

    return {'Title': title,
            'Author': author,
            'Date': date,
            'Text': raw_text,
            'Image': image}


def scrape(url, write=False):

    domain = urlparse(url).netloc.split('.')[1]
    if domain == 'npr':
        return npr_scrape(url)
    elif domain == 'cnn':
        return cnn_scrape(url)

    article = Article(url)

    article.download()
    article.parse()

    if write:
        modified_title = '-'.join(article.title.lower().split(' '))
        with open(f'scraped-text/{modified_title}.txt', 'w') as f:
            f.write(article.text)

    return {'Title': article.title,
            'Authors': article.authors,
            'Date': article.publish_date,
            'Text': article.text,
            'Image': article.top_image}


if __name__ == '__main__':
    test_url = 'https://www.cnn.com/2020/07/23/health/shutdown-us-contain-coronavirus-wellness/index.html'
    scrape_output = scrape(test_url)
    print(f'Title: {scrape_output["Title"]}')
    print(f'Authors: {scrape_output["Author"]}')
    print(f'Date: {scrape_output["Date"]}')
    print(f'Text: {scrape_output["Text"]}')
    print(f'Image: {scrape_output["Image"]}')