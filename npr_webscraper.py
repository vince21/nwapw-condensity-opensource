import requests
from bs4 import BeautifulSoup


def scrape(url):
    """
    Takes a URL for an NPR article and returns the text and images.
    :param url: NPR article link
    :type url: str
    :return: List of text elements in body and list of image links
    :rtype: ([bs4.element.Tag], [str]))
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    title = soup.find('div', class_='storytitle').find('h1').text.strip()

    author = soup.find('a', rel='author').text.strip()

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

    output_dict = {
                   'Title': title,
                   'Author': author,
                   'Text Elements': text_elements,
                   'Raw Text': '\n'.join([tag.text.strip() for tag in text_elements]),
                   'Images': images
                   }
    return output_dict


if __name__ == '__main__':
    test_url = 'https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans'
    scrape_output = scrape(test_url)
    print(f'Title: {scrape_output["Title"]}')
    print(f'Author: {scrape_output["Author"]}')
    print(f'Raw Text: {scrape_output["Raw Text"]}')
    print(f'Images: {scrape_output["Images"]}')
