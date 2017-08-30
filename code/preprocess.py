import numpy

from collections import deque
from collections import defaultdict

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

import re


#LANGUAGE = 'english'
LANGUAGE = 'italian'
REMOVE_STOPWORDS = True
SPLIT_PUNCTUATION = False
NEG_WORDS = {'no', 'not', 'nor'}
PUNCT = {',', '.', '?', ';', ':'}
BLANKS = {'__________', '_________', '___________', '____', '__________', '_______'}
STOPWORDS = set(stopwords.words(LANGUAGE))  # | BLANKS  # | PUNCT - BLANKS  # - NEG_WORDS

DOT_RGX = r'\.([a-zA-Z])'
MIN_RGX = r'([a-zA-Z])\-([a-zA-Z])'
NORM_RGX = r'[^A-Za-z0-9]'


def split_punctuation(string):

    string = re.sub(DOT_RGX, r'. \1', string)
    string = re.sub(MIN_RGX, r'\1 - \2', string)
    return string


def normalize_text(string):
    string = re.sub(NORM_RGX, ' ', string)
    return string


def alpha_part(token):
    valids = re.sub(r"[^A-Za-z]+", '', token)
    return valids


def filter_alpha_tokens(tokens):
    alpha_tokens = []
    for t in tokens:
        a_t = alpha_part(t)
        if a_t:
            alpha_tokens.append(a_t)
    return alpha_tokens


def remove_stoptokens(tokens):
    return [t.lower() for t in tokens if t.lower() not in STOPWORDS]


def pos_tag_tokens(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    return [w + '-' + t for w, t in tagged_tokens]


def set_stopwords(stopword_list):
    STOPWORDS.clear()
    for w in stopword_list:
        STOPWORDS.add(w)
    return STOPWORDS


def tokenize_sentence(sentence,
                      remove_stopwords=False,
                      split_punct=False,
                      add_pos_tags=False):

    if split_punct:
        sentence = split_punctuation(sentence)

    tokens = word_tokenize(sentence.lower())

    if remove_stopwords:
        tokens = remove_stoptokens(tokens)

    if add_pos_tags:
        tokens = pos_tag_tokens(tokens)

    return tokens


def tokenize_corpus(corpus):
    return [tokenize_sentence(d) for d in corpus]


def whitespace_tokenize(sentence):
    return sentence.strip().split()


def lemmatize_tokens(tokens, lemmatizer=None, add_pos_tags=False):

    if not lemmatizer:
        lemmatizer = WordNetLemmatizer()

    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    if add_pos_tags:
        lemmas = pos_tag_tokens(lemmas)

    return lemmas


def get_lemmatizer():
    return WordNetLemmatizer()


def tokenize_and_lemmatize(sentence):
    return lemmatize_tokens(tokenize_sentence(sentence))


def tokenize_and_stem(sentence):
    return stem_tokens(tokenize_sentence(sentence))


def tokenize_lemmatize_and_stem(sentence):
    return stem_tokens(lemmatize_tokens(tokenize_sentence(sentence)))


def get_stemmer(stemmer='snowball'):

    if stemmer == 'porter':
        return PorterStemmer()
    elif stemmer == 'lancaster':
        return LancasterStemmer()
    elif stemmer == 'snowball':
        return SnowballStemmer(LANGUAGE)
    else:
        raise ValueError('Unrecognized stemmer', stemmer)


def stem_tokens(tokens, stemmer=None, stemmer_type='snowball'):

    if not stemmer:
        stemmer = get_stemmer(stemmer_type)

    return [stemmer.stem(t) for t in tokens]


def stringify_tokens(tokens, sep=' '):
    return sep.join(tokens)


def process_sentence(sentence,
                     stemmer=None,
                     lemmatizer=None):

    tokens = tokenize_sentence(sentence)

    if stemmer:
        tokens = stem_tokens(tokens, stemmer)

    if lemmatizer:
        tokens = lemmatize_tokens(tokens, lemmatizer)

    processed_sentence = stringify_tokens(tokens)

    return processed_sentence


def process_sentences(sentences, stemmer=None, lemmatizer=None):
    return [process_sentence(s, stemmer, lemmatizer) for s in sentences]


SENT_TOK_MODES = {'sentence', 'texttile'}

PIPELINE_SEP = '>'

PREPROCESS_FUNC_DICT = {
    'id': lambda x: x,
    'nt': normalize_text,
    'ws': whitespace_tokenize,
    'sp': split_punctuation,
    'tk': tokenize_sentence,
    'rs': remove_stoptokens,
    'pt': pos_tag_tokens,
    'lm': lemmatize_tokens,
    'st': stem_tokens,
    'at': filter_alpha_tokens}


def wrap_preprocess(accum, f):
    return lambda x: f(accum(x))


def preprocess_factory(pipeline_string):

    steps = pipeline_string.split(PIPELINE_SEP)

    preprocess_func = lambda x: PREPROCESS_FUNC_DICT[steps[0]](x)

    for step in steps[1:]:
        preprocess_step = PREPROCESS_FUNC_DICT[step]
        preprocess_func = wrap_preprocess(preprocess_func, preprocess_step)

    return preprocess_func


def segment_into_paragraphs(text, mode='sentence'):
    """
    Segment one string of text into paragraphs
    """

    if mode == 'sentence':
        return sent_tokenize(text)
    elif mode == 'texttile':
        w = 20
        k = 4
        tokenizer = TextTilingTokenizer(w=w, k=k)
        return tokenizer.tokenize(text)

    else:
        raise ValueError('Mode {0} not in {1}'.format(mode, SENT_TOK_MODES))


BLACK_SUBSTRING = '__'


def sentence_contains_blank(sentence):
    return BLACK_SUBSTRING in sentence


def filter_sentences_with_blank(sentences, ids=False):
    if ids:
        return [(id, s) for id, s in sentences if sentence_contains_blank(s)]
    else:
        return [s for s in sentences if sentence_contains_blank(s)]


def filter_out_sentences_with_blank(sentences, ids=False):
    if ids:
        return [(id, s) for id, s in sentences if not sentence_contains_blank(s)]
    else:
        return [s for s in sentences if not sentence_contains_blank(s)]

RULE_IDS_TO_FILTER = {'ALL_OF_THE',
                      'EN_A_VS_AN',
                      'EACH_EVERY_NNS',
                      'THIS_NNS',
                      'SOME_OF_THE',
                      'PERIOD_OF_TIME',
                      'THE_FALL_SEASON',
                      'EVERYDAY_EVERY_DAY',
                      'ENGLISH_WORD_REPEAT_BEGINNING_RULE',
                      'ADVERB_WORD_ORDER',
                      'MASS_AGREEMENT',
                      'REASON_IS_BECAUSE',
                      'IN_A_X_MANNER',
                      'TRY_AND',
                      'SENTENCE_WHITESPACE',
                      'DID_BASEFORM',
                      'SENTENCE_FRAGMENT',
                      'MORFOLOGIK_RULE_EN_US',  # 301
                      'AGREEMENT_SENT_START',
                      'WHITESPACE_RULE',
                      'SO_THEREFORE',
                      'A_PLURAL',
                      'APART_A_PART',
                      'RATHER_THEN',
                      'HE_VERB_AGR',
                      'EN_QUOTES',
                      'IT_VBZ',
                      'LARGE_NUMBER_OF',
                      'THE_SUPERLATIVE',
                      'WHETHER',
                      'ADJECTIVE_IN_ATTRIBUTE',
                      'USE_TO_VERB',
                      'POSSESIVE_APOSTROPHE',
                      'NUMEROUS_DIFFERENT'}

LESS_RULE_IDS_TO_FILTER = {'IS_SHOULD',
                           'ALLOW_TO',
                           # 'ENGLISH_WORD_REPEAT_RULE',  # 351
                           'KIND_OF_A',
                           'ADMIT_ENJOY_VB',
                           'THERE_RE_MANY'
                           'AN_ANOTHER',
                           'NOW',
                           'DONT_NEEDS',
                           'MOST_SOME_OF_NNS',
                           'UPPERCASE_SENTENCE_START',
                           'CD_NN',  # 9
                           'HAVE_PART_AGREEMENT',  # 18
                           'PHRASE_REPETITION',  # 29
                           'THEIR_IS',  # 9
                           'THE_HOW',  # 2
                           'FROM_FORM',  # 18
                           # 'DT_DT',# 373
                           'A_UNCOUNTABLE',  # 19
                           'MAY_COULD_POSSIBLY',  # 3
                           'IT_IS',  # 17
                           'THERE_S_MANY',  # 1
                           'A_RB_NN',  # 10
                           'NON3PRS_VERB',  # 7
                           'MODAL_OF',  # 1
                           'DOES_X_HAS',  # 10
                           'AN_THEN',  # 2
                           'DT_PRP',  # 39
                           'DT_JJ_NO_NOUN',  # 3
                           # 'EN_UNPAIRED_BRACKETS', # 72
                           'IS_WERE',  # 2
                           # 'A_INFINITVE' # 110
                           'BEEN_PART_AGREEMENT',  # 41
                           'ONE_PLURAL',  # 3
                           'DOES_NP_VBZ',  # 8
                           'MANY_NN',  # 4
                           'BE_CAUSE',  # 1
                           'MANY_NN_U',  # 11
                           'NOUN_AROUND_IT',  # 1
                           'IF_IS',  # 1
                           'TO_NON_BASE',  # 8
                           }
# import language_check


def get_syntax_rules_to_filter(correct_sentences, tool=None, verbose=False):

    if not tool:
        tool = language_check.LanguageTool('en-US')

    rules_to_filter = set()

    for s in correct_sentences:
        matches = tool.check(s)

        if matches:
            for m in matches:
                rules_to_filter.add(m.ruleId)

    return rules_to_filter


def is_sentence_syntax_correct(sentence,
                               tool=None,
                               rule_filters=RULE_IDS_TO_FILTER,
                               verbose=False):

    if not tool:
        tool = language_check.LanguageTool('en-US')

    matches = tool.check(sentence)

    real_matches = []

    for m in matches:
        if m.ruleId not in rule_filters:
            real_matches.append(m)

    if verbose:
        print('matches', matches)
        print('after filters:', real_matches)

    return len(real_matches) == 0, matches


def check_sentences_syntax(sentences,
                           tool=None,
                           ids=False,
                           rule_filters=RULE_IDS_TO_FILTER,
                           verbose=False):

    from language_check import LanguageTool
    LanguageTool._TIMEOUT = 10

    if not tool:
        tool = language_check.LanguageTool('en-US')

    sentences_to_process = deque(sentences)

    n_sentences = len(sentences)
    n_correct = 0
    i = 0

    checks = []
    matches = defaultdict(list)
    id_list = []

    while sentences_to_process:
        try:
            print('Processing sentence {0}, correct {1}/{2}'.format(i,
                                                                    n_correct,
                                                                    n_sentences),
                  end='    \r')
            current_sentence = sentences_to_process.popleft()

            if ids:
                current_id, current_sentence = current_sentence

            check, model = is_sentence_syntax_correct(current_sentence,
                                                      tool=tool,
                                                      rule_filters=rule_filters,
                                                      verbose=verbose)

            checks.append(check)
            for m in model:

                # if m.ruleId in LESS_RULE_IDS_TO_FILTER:
                #     print('\n', m.ruleId, current_sentence)

                if ids:
                    matches[m.ruleId].append(current_id)
                else:
                    matches[m.ruleId].append(i)

            if check:
                n_correct += 1
            else:
                if ids:
                    id_list.append(current_id)

            i += 1
        except:
            print('An error occurred, retrying for {}'.format(i), current_sentence)

            if ids:
                current_sentence = (current_id, current_sentence)
            sentences_to_process.append(current_sentence)
            tool = language_check.LanguageTool('en-US')

    return checks, matches, id_list


def get_common_tokens(sentences, as_list=False):
    """
    sentences is a sequence of tokenized sentences
    """
    common_tokens = set.intersection(*[set(s) for s in sentences])

    if as_list:
        common_tokens = list(common_tokens)

    return common_tokens


def remove_common_tokens(sentence, other_sentences):
    return [[t for t in o if t not in sentence] for o in other_sentences]


def get_token_residual(sentence, common_part):
    return [t for t in sentence if t not in common_part]


FIG_STR = '( Figure below )'

CHARS_TO_REPLACE = {'°', '’', '“', '”', '’', '—', '/', '(', ')', '→', '↓', 'Ω'}

SUB = ' '

REP = dict((re.escape(k), SUB) for k in CHARS_TO_REPLACE)

CHARS_RGX = re.compile("|".join(REP.keys()))

NUM_RGX = '[+-]?\d+|(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
NUM_RGX = re.compile(NUM_RGX)


def clean_text_from_numbers(text, sub=SUB):
    return NUM_RGX.sub(sub, text)


def clean_text_from_chars(text):
    return CHARS_RGX.sub(lambda m: REP[re.escape(m.group(0))], text)


def clean_text_from_non_ascii(text):
    # return bytes(text.strip(), 'utf-8').decode("ascii", "ignore").encode('ascii')
    return ''.join(i for i in text if ord(i) < 128)


def clean_text_from_captions(text):
    return text.replace(FIG_STR, SUB)

# URL_RGX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
URL_RGX = '(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'
URL_RGX = re.compile(URL_RGX)


def clean_text_from_urls(text, sub=SUB, pattern=URL_RGX):
    return pattern.sub(sub, text)


LATEX_RGX = re.compile(r"^\\[^\s]*", re.MULTILINE)


def clean_text_from_latex(text, sub=SUB):
    return re.sub(LATEX_RGX, sub, text)


def clean_text(text,
               clean_chars=True,
               clean_captions=True,
               clean_non_ascii=True,
               clean_urls=True,
               clean_nums=True,
               clean_latex=True):

    if clean_non_ascii:
        text = clean_text_from_non_ascii(text)

    if clean_latex:
        text = clean_text_from_latex(text)

    if clean_chars:
        text = clean_text_from_chars(text)

    if clean_captions:
        text = clean_text_from_captions(text)

    if clean_urls:
        text = clean_text_from_urls(text)

    if clean_nums:
        text = clean_text_from_numbers(text)

    return text


def combine_all_others(seq, pos, sep):
    """
    Returns a concatenation of all elements of a sequence except for
    one specified by pos
    """
    return sep.join(list(seq[:pos]) + list(seq[pos + 1:]))

NEGATION_TOKENS = {'not', 'non'}
EXCEPT_TOKENS = {'except'}
ALL_ABOVE_SUBSTR = 'all of the above'


def get_sentences_by_token(questions, tokens):

    sentences_ids = numpy.zeros(len(questions), dtype=bool)

    for i, q in enumerate(questions):
        for w in q:
            if w.lower() in tokens:
                sentences_ids[i] = True
                break

    return sentences_ids


def get_negation_questions(dataset_matrix, negations=NEGATION_TOKENS):
    questions = dataset_matrix[:, 0]
    questions_tokens = tokenize_corpus(questions)
    return get_sentences_by_token(questions_tokens, negations)


def get_except_questions(dataset_matrix, excepts=EXCEPT_TOKENS):
    questions = dataset_matrix[:, 0]
    questions_tokens = tokenize_corpus(questions)
    return get_sentences_by_token(questions_tokens, excepts)


def get_all_above_questions(dataset_matrix, pattern=ALL_ABOVE_SUBSTR):
    answers = dataset_matrix[:, 1:]

    sentences_ids = numpy.zeros((len(answers), 4), dtype=bool)

    for i, ans_i in enumerate(answers):
        for j, a in enumerate(ans_i):
            if pattern in a.lower():
                sentences_ids[i, j] = True

    return sentences_ids


def transform_answers_for_negations(answers, sep=' '):
    """
    from [A, B, C, D] to [B+C+D, A+C+D, A+B+D], A+B+C
    """

    proc_answers = [combine_all_others(answers, i, sep) for i, ans in enumerate(answers)]
    return proc_answers


def transform_answers_for_all_of_above(answers, all_pos=3, sep=' '):
    """
    from [A, B, C, All of the above] to [A, B, C, A+B+C]
    """
    proc_answers = answers[:]
    proc_answers[all_pos] = combine_all_others(answers[:all_pos + 1], all_pos, sep)
    return proc_answers


def transform_dataset_answers(dataset_matrix,
                              negative_ids,
                              except_ids,
                              all_above_ids,
                              sep=' '):

    n_questions = dataset_matrix.shape[0]

    for i in range(n_questions):
        answers_i = dataset_matrix[i, 1:]

        if all_above_ids is not None:
            answer_all_ids = all_above_ids[i]
            for j, a in enumerate(answer_all_ids):
                if a == 1:
                    dataset_matrix[i, 1:] = \
                        numpy.array(transform_answers_for_all_of_above(answers_i,
                                                                       j,
                                                                       sep))
                    break
        elif negative_ids is not None and negative_ids[i]:
            dataset_matrix[i, 1:] = numpy.array(transform_answers_for_negations(answers_i, sep))
        elif except_ids is not None and except_ids[i]:
            dataset_matrix[i, 1:] = numpy.array(transform_answers_for_negations(answers_i, sep))

    return dataset_matrix
