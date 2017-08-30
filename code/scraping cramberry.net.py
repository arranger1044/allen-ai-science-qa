
# coding: utf-8

# In[16]:

#
# http://cramberry.net/sets/6419-biology

import numpy
from selenium import webdriver

from bs4 import BeautifulSoup

import time
from time import perf_counter

import urllib
import requests

from pyvirtualdisplay import Display

PHANTOMJS_PATH = '/usr/bin/phantomjs'

NOTEBOOK = False


# In[2]:

def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


# In[35]:

def get_flashcards_text(page_url,
                        waiting_time=10,
                        click_waiting_time=0.5,
                        close=False):

    card_texts = []
    get_start_t = perf_counter()
    #
    # making the request
    print(page_url)
    html_source = requests.get(page_url).content
    soup = BeautifulSoup(html_source, 'lxml')
    get_end_t = perf_counter()

    print('\t\t\tgot request in {} secs'.format(get_end_t - get_start_t), end='       \r')

    proc_start_t = perf_counter()

    card_table = soup.find('table', {'class': 'table table-striped'})

    if card_table:
        rows = card_table.findChildren('tr')
        for row in rows:
            cells = row.findChildren('td')
            #
            # the third column is wrong
            # for cell in cells[:2]:
            card_texts.extend(reversed([cell.string for cell in cells[:2]]))

    qas = list(pairwise(card_texts))

    proc_end_t = perf_counter()
    print('Processed {0} cards in {1}'.format(
        len(qas), proc_end_t - proc_start_t), end='       \r')

    return qas


# In[36]:

def save_qa_list_to_file(file_path,
                         qa_list,
                         separator='\t'):
    with open(file_path, 'w') as qa_file:
        for q, a in qa_list:
            qa_line = q + separator + a + '\n'
            qa_file.write(qa_line)


# In[37]:

if NOTEBOOK:
    url_ex = 'https://cramberry.net/sets/6419-biology'
    qas = get_flashcards_text(url_ex)
    for q, a in qas:
        print('Q:', q)
        print('A:', a)
    save_qa_list_to_file('qa-test.tsv', qas)


# In[49]:

BASE_URL = 'https://cramberry.net'


def scrape_flashcard_url_list(page_url,
                              close=False):

    get_start_t = perf_counter()
    #
    # making the request

    html_source = requests.get(page_url).content
    soup = BeautifulSoup(html_source, 'lxml')
    get_end_t = perf_counter()

    print('\t\t\tgot request in {} secs'.format(get_end_t - get_start_t), end='       \r')

    #
    # retrieving hrefs
    card_links = soup.findAll('a', {'class': 'deck'})
    flashcards_links = [BASE_URL + card['href'] for card in card_links]

    return flashcards_links


# In[51]:

if NOTEBOOK:
    url_ex = 'https://cramberry.net/sets/all?utf8=%E2%9C%93&q=chem&commit=Search&page=1'
    fls = scrape_flashcard_url_list(url_ex)
    for f in fls:
        print(f)


# In[53]:

from collections import deque
import os

EXT = '.tsv'


def scrape_flashcards_from_page(page_url,
                                page_id,
                                output_dir='../data/cramberry.net/flashcards/Astronomy/',
                                card_content_id='currentCard',
                                phantomjs_path=PHANTOMJS_PATH,
                                waiting_time=2):

    #
    # retrieve the url for each flashcard from the page
    url_list = scrape_flashcard_url_list(page_url)
    print('There are {} decks'.format(len(url_list)))

    #
    # saving them in an index
    output_dir = output_dir + '{0}/'.format(page_id)
    os.makedirs(output_dir, exist_ok=True)
    index_path = output_dir + '{0}.index'.format(page_id)
    with open(index_path, 'w') as index_file:
        for url in url_list:
            index_file.write(url + '\n')

    urls_to_process = deque(url_list)
    n_urls = len(url_list)
    i = 0

    page_start_t = perf_counter()

    while urls_to_process:

        try:
            #
            # take the first one
            flashcard_url = urls_to_process.popleft()
            flashcard_id = flashcard_url.split('/')[-1]
            print('- [{0}/{1}@{3}] Processing flashcard {2}'.format(i + 1,
                                                                    n_urls,
                                                                    flashcard_id,
                                                                    page_id))

            #
            # get path and check that it has not been processed yet
            output_path = output_dir + flashcard_id + EXT
            if not os.path.exists(output_path):

                #
                # get qa
                q_start_t = perf_counter()
                qas = get_flashcards_text(flashcard_url)
                q_end_t = perf_counter()
                print('flashcard extracted in {}'.format(q_end_t - q_start_t))

                if qas:
                    #
                    # saving
                    save_qa_list_to_file(output_path, qas)
                    print('\tsaved to file {}'.format(output_path), end='            \r')
                else:
                    print('Empty set', end='            \r')

            else:
                print('File already exists, skipping')
            i += 1

        except Exception as e:

            print('\t**** [{0}] Error while processing {1}, retrying ****'.format(e,
                                                                                  flashcard_url))
            urls_to_process.appendleft(flashcard_url)

    page_end_t = perf_counter()
    print('\n!!!! page completed in {} secs !!!!'.format(page_end_t - page_start_t))


# In[54]:

if NOTEBOOK:
    url = 'https://cramberry.net/sets/all?utf8=%E2%9C%93&q=biology&commit=Search&page=1'
    output_path = '../data/cramberry.net/flashcards/'
    scrape_flashcards_from_page(url, 1, output_path)


# In[ ]:
def generate_url_pages_list(base_url, min_page_num, max_page_num):
    return [base_url + str(i) for i in range(min_page_num, max_page_num)]

import argparse

BASE_URL_PAGE = 'https://cramberry.net/sets/all?utf8=%E2%9C%93&q=astronomy&commit=Search&page='
OUT_PATH = '../data/cramberry.net/flashcards/Astronomy/'
N_PROCESSES = 1
MIN_PAG = 1
MAX_PAG = 10

if __name__ == '__main__':
    #
    # creating the parser for input args
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str,
                        default=OUT_PATH,
                        help='output directory')

    parser.add_argument('-f', '--first-page', type=int,
                        default=MIN_PAG,
                        help='number of page to start from')

    parser.add_argument('-l', '--last-page', type=int,
                        default=MAX_PAG,
                        help='number of page to stop to')

    parser.add_argument('-u', '--base-url', type=str,
                        default=BASE_URL_PAGE,
                        help='base url')

    # parsing the args
    args = parser.parse_args()

    #
    # generating page urls
    pages_urls = generate_url_pages_list(args.base_url, args.first_page, args.last_page + 1)

    #
    # starting scraping
    for i, url in zip(range(args.first_page, args.last_page + 1), pages_urls):
        print('\n\n\nProcessing page: {}\n\n\n'.format(url))
        scrape_flashcards_from_page(url, i, output_dir=args.output)
