
# coding: utf-8

# In[1]:

# http://www.studydroid.com/index.php?page=viewPack&packId=1571
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


# In[64]:

def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def text_with_spaces(elem):
    text = ''
    for e in elem.recursiveChildGenerator():
        if isinstance(e, str):
            text += e.strip()
        elif e.name == 'br':
            text += ' '
    return text


def filter_blank(cards):
    return [(q, a) for q, a in cards if q and a]


def get_flashcards_text(page_url):

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

    cards = soup.findAll('tr', {'align': 'left'})

    if cards:
        for card in cards:
            if card.get('style') is None or not 'display: none;' in card.get('style'):
                for front in card.findAll('div'):
                    c_text = text_with_spaces(front)
                    c_text = c_text.replace('\n', ' ').replace('\r', '')
                    card_texts.append(c_text)

    qas = list(pairwise(card_texts))

    qas = filter_blank(qas)

    proc_end_t = perf_counter()
    print('Processed {0} cards in {1}'.format(
        len(qas), proc_end_t - proc_start_t), end='       \r')

    return qas


# In[65]:

def save_qa_list_to_file(file_path,
                         qa_list,
                         separator='\t'):
    with open(file_path, 'w') as qa_file:
        for q, a in qa_list:
            qa_line = q + separator + a + '\n'
            qa_file.write(qa_line)


# In[67]:

if NOTEBOOK:
    url_ex = 'http://www.studydroid.com/printerFriendlyViewPack.php?packId=1571'
    qas = get_flashcards_text(url_ex)
    print(len(qas))
    for q, a in qas:
        print('Q:', q)
        print('A:', a)
    save_qa_list_to_file('qa-test.tsv', qas)


# In[71]:

BASE_URL = 'http://studydroid.com/'


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
    card_rows = soup.findAll('tr', {'class': ['row_light', 'row_dark']})
    #
    # skipping header
    flashcards_links = []
    if card_rows:
        for row in card_rows[1:]:
            tds = [td for td in row.findAll('td')]
            if tds:
                flashcards_links.append(BASE_URL + tds[0].p.a['href'])

    return flashcards_links


# In[73]:

if NOTEBOOK:
    url_ex = 'http://www.studydroid.com/index.php?page=search&search=science&x=0&y=0&begin=2500'
    fls = scrape_flashcard_url_list(url_ex)
    print(len(fls))
    for f in fls:
        print(f)


# In[74]:

from collections import deque
import os

EXT = '.tsv'

PRINT_URL = 'http://www.studydroid.com/printerFriendlyViewPack.php?'


def scrape_flashcards_from_page(url_list,
                                page_id,
                                output_dir,
                                waiting_time=2):

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
            flashcard_id = flashcard_id.split('?')[-1]
            flashcard_id = flashcard_id.split('&')[-1]

            flashcard_url = PRINT_URL + flashcard_id
            print('- [{0}/{1}@{3}] Processing flashcard {2}'.format(i + 1,
                                                                    n_urls,
                                                                    flashcard_url,
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


# In[ ]:

OFFSET_STR = '&begin='


def scrape_flashcards_from_pages(page_url,
                                 n_pages,
                                 cards_per_page=25,
                                 output_dir='../data/studydroid.com/flashcards/Science/',
                                 waiting_time=2):

    for i in range(n_pages):
        begin_offset = cards_per_page * i

        print('Trying to process with offset', begin_offset)

        offset_page_url = page_url + OFFSET_STR + str(begin_offset)

        #
        # retrieve the url for each flashcard from the page
        url_list = scrape_flashcard_url_list(offset_page_url)

        if url_list:
            print('There are {} decks'.format(len(url_list)))

            #
            # saving them in an index
            output_dir_i = output_dir + '{0}/'.format(i + 1)
            os.makedirs(output_dir_i, exist_ok=True)
            index_path = output_dir_i + '{0}.index'.format(i + 1)
            with open(index_path, 'w') as index_file:
                for url in url_list:
                    index_file.write(url + '\n')

            scrape_flashcards_from_page(url_list,
                                        i + 1,
                                        output_dir=output_dir_i)
        else:
            print('No more cards, breaking.')
            break

import argparse

BASE_URL_PAGE = 'http://www.studydroid.com/index.php?page=search&x=0&y=0&search='
OUT_PATH = '../data/studydroid.com/flashcards/'
N_PROCESSES = 1
MAX_PAG = 100

if __name__ == '__main__':
    #
    # creating the parser for input args
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str,
                        default=OUT_PATH,
                        help='output directory')

    parser.add_argument('-n', '--max-n-pages', type=int,
                        default=MAX_PAG,
                        help='max number of pages')

    parser.add_argument('-c', '--cards-per-page', type=int,
                        default=26,
                        help='num cards per page')

    parser.add_argument('-t', '--tags', type=str,
                        nargs='+',
                        default=['Science', 'Biology', 'Physics',
                                 'Chemistry', 'Anatomy', 'Physiology', 'Cell',
                                 'Chem', 'Earth'],
                        help='output directory')

    parser.add_argument('-u', '--base-url', type=str,
                        default=BASE_URL_PAGE,
                        help='base url')

    # parsing the args
    args = parser.parse_args()

    for tag in args.tags:

        print('\n\n*** Processing tag {} ****\n\n'.format(tag))

        #
        # generating page urls
        pages_url = BASE_URL_PAGE + tag

        output_dir = args.output + tag + '/'
        #
        # starting scraping
        print('\n\n\nProcessing page: {}\n\n\n'.format(pages_url))
        scrape_flashcards_from_pages(pages_url, args.max_n_pages,
                                     cards_per_page=args.cards_per_page,
                                     output_dir=output_dir)
