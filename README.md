# Allen AI Science Question Answering Challenge

Code for the [5th place solution](https://www.kaggle.com/c/the-allen-ai-science-challenge/leaderboard) by team "_A Pure Logical Approach_" for the [Allen AI Science QA Kaggle Challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge)

## Architecture overview

Our architecture is a layered stacked architecture composed by several components. We will briefly review them in the following parts.

The first layer is made by the predictors we extract by using several combinations of Information Retrival parameters on top of different combinations of corpora which we indexed using Lucene.
With Lucene we considered these parameters:

- _**similarity metrics involved**_: VSM, BM25, LanguageModel, DivergenceFromRandomness
- _**analyzers**_: Standard Analyzer, English Analyzer

We executed queries composed by a question plus one answer concatenated, with one of the possible models from the combinations of the above parameters.
Then, we started considering the top k retrieved documents, with k going from 1 up to 40. For each value of k we summed the scores obtained by Lucene for the top k documents, creating a confidence score (a predictor) for the considered answer. We repeated this process for all the answers and we then normalized our scores for all four answers such to have some probability predictors ranging from 0 to 1 and summing to 1.

Moreover, we considered two alternative ways to perform a query, by including a form of negation or not. Considering that only one of the possible answers is the true one, while performing a query, if the negation option is considered, we are negating in Lucene all the tokens coming from the other answers that do not appear in the currently considered one. In this way we are trying to focus only on those part of answers that are highly discriminative in the retrieval phase. We found out that for many similarities on different corpora this lead to an accuracy improvement. When the accuracy did not improve we ended up with another set of predictors in the mix.

If we used all the possible combinations above (8), turned the negation parameter on or off (2), with k up to 40, we ended up with 640 predictors for each dataset, given a corpus. In practice we used in end much less predictors, exploring the variations with k up to 10 since we noticed that for k > 10 the predictors were highly correlated.


In the end, our architecture is very similar in spirit to the one by user Cardal who won the competition. [_**Details make a huge difference**_](https://github.com/Cardal/Kaggle_AllenAIscience)!

## Requirements

We have run all our scripts for building our architecture on different
machines. However, here we are defining the minimum requirements for
our code to run properly.

### Hardware requirements:
- a server machine with:
    - 16 Gb of ram
    - the more cores the better (we parallelized up to 8 threads for  w2v and xgboost)

### Software requirements:

- a Linux machine (we used Ubuntu 14.04+)
- for YATStool:
   - Java 8
   - other libraries have been included in the jar

- for python:
    - python 3.4+ (we are using iPython 3.2.1 as the interpreter)
    - numpy 1.10+
    - scipy 0.16+
    - sklearn 0.17+
    - pandas 0.15+
    - nltk 3.1+:
      	- stopwords package
	    - punkt
	    - snowball
    - gensim 0.12.3
    - selenium 2.48 (for scraping)
    - beautifulsoup 4.4.1 (for scraping)
    - pyvirtualdisplay (for scraping)
    - seaborn 0.6+ (for plotting options)
    - matplotlib 1.5.dev

- Additional tools:
    - phantomjs (to use with selenium for scraping)
    - xgboost (downloaded as the latest version from the git repo)


## Corpora used
We used several corpora to build up our architecture. We exploited three different kinds of resources:
- science books
- science flashcards
- a filtered version of wikipedia

Concerning science books we used:

- ck12.org books: We scraped all the books reachable from this url [http://www.ck12.org/standards/](http://www.ck12.org/standards/) (with the exception of [http://www.ck12.org/book/CK-12-Algebra-I-Second-Edition/](http://www.ck12.org/book/CK-12-Algebra-I-Second-Edition/'), [http://www.ck12.org/book/CK-12-Geometry-Basic/](http://www.ck12.org/book/CK-12-Geometry-Basic/), [http://www.ck12.org/book/CK-12-Basic-Probability-and-Statistics-A-Short-Course/](http://www.ck12.org/book/CK-12-Basic-Probability-and-Statistics-A-Short-Course/)) and we build a unified corpus, from here on 'ck12'

- a summary of ck12.org topics, called 'concepts' from here on, downloaded from this url [http://www.ck12.org/flx/show/epub/user%3AanBkcmVjb3VydEBnbWFpbC5jb20./Concepts_b_v8_vdt.epub](http://www.ck12.org/flx/show/epub/user%3AanBkcmVjb3VydEBnbWFpbC5jb20./Concepts_b_v8_vdt.epub) in a epub format, converted to simple text using Calibre (https://calibre-ebook.com/)

- science books downloaded from [http://www.openculture.com/free_textbooks](http://www.openculture.com/free_textbooks) ('openculture') from pdf format and converted with pure txt with pdf2txt.py (from pdf miner [http://www.unixuser.org/~euske/python/pdfminer/](http://www.unixuser.org/~euske/python/pdfminer/)), namely:
    - [http://www.learner.org/courses/biology/textbook/index.html](http://www.learner.org/courses/biology/textbook/index.html)
    - [http://www.curriki.org/xwiki/bin/view/Coll_Group_CLRN-OpenSourceEarthScienceCourse/EarthSystemsAnEarthScienceCourse?bc=](http://www.curriki.org/xwiki/bin/view/Coll_Group_CLRN-OpenSourceEarthScienceCourse/EarthSystemsAnEarthScienceCourse?bc=)
    - [https://archive.org/details/ost-physics-essentialphysics1](https://archive.org/details/ost-physics-essentialphysics1)
    - [http://scipp.ucsc.edu/outreach/index2.html](http://scipp.ucsc.edu/outreach/index2.html)
  
- free pdf science books from fhsst (converted to pure text with pdfminer as well) available at this url [http://www.nongnu.org/fhsst/2_project.html](http://www.nongnu.org/fhsst/2_project.html), namely:
    - Chemistry
    - Physics
  
- science books downloaded from [https://www.openstaxcollege.org/books](https://www.openstaxcollege.org/books) (downloaded as ebook and converted with Calibri), namely:
    - Anatomy & Physiology
    - Biology
    - Concepts of Biology
    - Chemistry
    - Physics

- One free science book on chemistry downloaded from saylor.org

- free science books downloaded as pdfs from [http://www.schools.utah.gov/CURR/science/OER.aspx](http://www.schools.utah.gov/CURR/science/OER.aspx) (converted using pdfminer), namely:
    - 3rd Grade: Science
    - 4th Grade: Science
    - 5th Grade: Science
    - 6th Grade: Science
    - 7th Grade: Integrated Science
    - 8th Grade: Integrated Science
    - 8th Grade: Integrated Science (2014)
    - Biology
    - Chemistry
    - Earth Science
    - Physics


Concerning the flashcards, we exploited these free internet sites from which we scraped several decks of flashcards, recorded as pure text as pairs of question and answer:

- studystack.com, as we scraped the following categories:
  - Biology the first 5 pages, ordered by relevance
  - Chemistry the first 3 pages, ordered by relevance
  - Earth Science, the first 3 pages, ordered by relevance
  - Physical Science, the first 2 pages, ordered by relevance
  - Physics the first page, ordere by relevance
  - Science, the first 15 pages, ordered by relevance
  
- quizlet.com, where we scraped the following categories, taking the first 10 pages for each one:
  - Anatomy
  - Astronomy
  - Biology
  - Chemistry
  - Earth Science
  - Physics
  - Physiology
  - Science

- cramberry.net, where we got all the cards from these categories:
  - Anatomy (55 pages)
  - Astronomy (11 pages)
  - Biology (105 pages)
  - Chem (22 pages)
  - Chemistry (64 pages)
  - Earth Science (345 pages)
  - Physics (37 pages)
  - Physiology (32 pages)
  - Science (231 pages)

- cueflash.com, from which we took the flashcards from the following categories (all possible results up to 100 max pages):
  - Science
  - Biology,
  - Physics
  - Chemistry
  - Anatomy
  - Physiology
  - Cell
  - Chem
  - Earth

- studydroid.com from which we took the flashcards from the following categories (all possible results up to 100 max pages):
  - Science
  - Biology,
  - Physics
  - Chemistry
  - Anatomy
  - Physiology
  - Cell
  - Chem
  - Earth 

## Word2vec
We exploited word2vec (w2v) models as another predictor in our architecture. The output of the models is a similarity score between the question and one of the answers.
We used the python lib [gensim](https://radimrehurek.com/gensim/) to use w2v and implemented our routines in the script `word2vec_exp.py`.
There are several routines to run a grid search among w2v parameters and evaluate an existing model. Moreover one can specify different functions to preprocess the text (both question and answer), to compute the embedding for a sentence, the how to compute the similarity among sentences and which metric to use in doing so. We found out that the best combination consists of using a light pipeline processing consisting in separating punctuation, tokenization and common stopword removal ('st>tk>rs' value for the parameter '--proc-func'); then use the sum of the token embeddings to get an embedding for a whole sentence ('sum' value for the parameter '--ret-func'). We apply a version of the quantistic negation (implemented with Grahm Schmidt orthogonalization routines) to compute the similarity among a question 'q' and answer 'a_i'. First we get the embedding for question 'q', then for 'a' we get the embedding by doing the sum of its token embeddings, then we negate it with the embedding which is the result of summing the embeddings of all other tokens in the other possible answers that are not present in 'a_i'. The rationale behind this is that we want to exploit the fact that we are in an answer selection task and only one answer can be true at a time. In this way we are subtracting from 'a_i' embedding all the tokens embeddings that are coming from the 'wrong' answers and are not in 'a_i'. As a similarity metric we use the correlation coefficient instead of the classical cosine similarity since we have found it to be more robust when combined with Ghram-Schmidt.

We learned three different models based on three different corpora:
- ck12
- concepts
- studystack

We found that all other corpora were too scarce in vocabulary or too noisy and the w2v accuracy performances degraded a lot (< 39%, while on studystack they are at ~42%).
We are providing these three learned models in the directory 'model/w2v-scores/'. We will now show how to learn them and evaluate them again, however we were not able to set a seed properly in the library. It appears that this is an issue related on the multithreading randomicity (more threads are needed to process the whole corpora, however even with one thread alone the C implementation seed cannot be set properly) We opened an issue for the lib: [https://github.com/piskvorky/gensim/issues/546](https://github.com/piskvorky/gensim/issues/546)

An example to learn a w2v model from the ck12 corpus (after having it decompressed) and evaluate on a training set and on a test set (whose paths are to specify) is to run the following command:

    ipython -- word2vec_exp.py ../../corpora/ck12.txt -e 500  -w 20 -i 50 -j 8 --proc-func 'sp>tk>rs' --pred-func gra --ret-func sum --sim-func cor --alpha 0.025 -o ../../model/w2v-scores/ --exp-name ck12 --train <training-set-pat> --test <test-set-path>

this set all the best parameters we found and stores the result in a folder called exp_ck12 inside model/w2v_scores. Inside this folder the model is saved in 0/0.model as a binary format that is readable with gensim.

To run this command for all three corpora we have prepared another bash script: train_w2v.sh in which one has to specify only the training and test paths once:

    ./train_w2v ../../data/training_set.tsv  ../../validation_set.tsv

If one has an already learned model and just wants to evaluate on another pair of dataset, the same word2vec_exp script can be used, by specifying the model path and the dataset paths:

    ipython -- word2vec_exp.py <model-path>  --pred-func gra --ret-func sum --sim-func cor --eval -o ../../model/w2v-scores/ --exp-name ck12-eval --train <training-set-path> --test <test-set-path>

We are also providing a bash script that evaluates the previously learned models for all three corpora and produces the outputs as the predicted scores for the training and test set (the path of the models is fixed and assumed to be the one written in the above scripts)

    ./eval_w2v ../../data/training_set.tsv  ../../validation_set.tsv

After the evaluation, in model/w2v-scores/ new folders are created, e.g. ck12-eval for the scores on ck12.


## A Pure Logical Approach
Despite the name, we actually never employed some symbolic or even relational approach as one of our predictors.

The team was formed by:

- me (brau1589)
- [Gaetano Rossiello](http://www.di.uniba.it/~swap/index.php?n=Membri.Rossiello) (GaetanGate)
- [Pierpaolo Basile](http://www.di.uniba.it/~swap/index.php?n=Membri.Basile) (pierobak)
- [Annalina Caputo](https://annalina.github.io/) (logicleap)
- [Pasquale Minervini](https://github.com/pminervini) (d3adbeef)

