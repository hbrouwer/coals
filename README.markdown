COALS
=====

Synopsis
--------
[Correlated Occurrence Analogue to Lexical Semantics]
(http://www.cnbc.cmu.edu/~plaut/papers/pdf/RohdeGonnermanPlautSUB-CogSci.COALS.pdf)

Description
-----------

This implements the Correlated Occurrence Analogue to Lexical Semantics
(COALS; Rohde, Gonnerman, & Plaut, 2005). COALS represents word meaning by
means of high-dimensional vectors derived from word co-occurrence patterns
in large corporora. The construction of these vectors starts by compiling
a co-occurrence table using a ramped (or alternatively, flat) n-word window.
Rohde et al. use a 4-word, ramped window:

     1 2 3 4 [0] 4 3 2 1

Next, all but the m (14.000 in case of Rohde et al.) columns, reflecting the
most frequent words, are discarded. Co-occurrence counts are then converted
to word-pair correlations; negative correlations set to zero, and positive
correlations are replaced by their square root in order to reduce difference
between them. The rows of the co-occurrence table then represent COALS
vectors for their respective words. Optionally, Singular Value Decomposition
(SVD) can be used to reduce the dimensionality of these vectors, and these
reduced vectors can, in turn, be converted to binary vectors, by setting
negative components to zero, and positive components to one.

The implementation also supports a recent extension to the COALS model as
proposed by Chang et al. (2012).

* Chang, Y., Furber, S., & Welbourne, S. (2012). *Generating Realistic
  Semantic Codes for Use in Neural Network Models*. In Miyake, N., Peebles,
  D, and Cooper, R. P. (Eds.). Proceedings of the 34th Annual Meeting of the
  Cognitive Science Society (CogSci 2012).

* Rohde, D. L. T., Gonnerman, L. M., & Plaut, D. C. (2005). *An Improved
  Model of Semantic Similarity Based on Lexical Co-Occurrence*.

Disclaimer
----------

I released the code "as is"; that is, in the state in which I used it for
the modeling work in my [PhD
thesis](http://dissertations.ub.rug.nl/faculties/arts/2014/h.brouwer/?pLanguage=en).
As such, it does what it has to do, but there is still a lot of room for
improvement.

Usage
-----

    $ ./coals

    COALS version 0.99-beta
    Copyright (c) 2012-2014 Harm Brouwer <me@hbrouwer.eu>
    Center for Language and Cognition, University of Groningen
    Netherlands Organisation for Scientific Research (NWO)

    usage ./coals [options]

      constructing COALS vectors:
        --wsize <num>       set co-occurrence window size to <num>
        --wtype <type>      use <type> window for co-occurrences (dflt: ramped)
        --rows <num>        set number of rows of the co-occurrence matrix to <num>
        --cols <num>        set number of cols of the co-occurrence matrix to <num>
        --dims <num>        reduce COALS vectors to <num> dimensions (using SVD)
        --vtype <type>      construct <type> (real/binary[_pn]) vectors (dflt: real)
        --unigrams <file>   read unigram counts (word frequencies) from <file>
        --ngrams <file>     read n-gram counts (co-occurrence freqs.) from <file>
        --output <file>     write COALS vectors to <file>
        --enforce <file>    enforce inclusion of words in <file>

        --pos_fts <num>     number of positive features (for binary_pn vectors)
        --neg_fts <num>     number of negative features (for binary_pn vectors)

      extracting similar words:
        --vectors <file>    compute similarities on basis of vectors in <file>
        --output <file>     write top-k similar word sets to <file>
        --topk <num         extract top-<num> similar words for each word

      basic information for users:
        --help              shows this help message
        --version           shows version

Example
-------

Here is an example of how to create 100-bits binary COALS vectors for the
15000 most frequent words in a corpus, using a 4-word ramped window, and the
14000 most frequent features:

    coals --wsize 4 --wtype ramped --rows 15000 --cols 14000
          --unigrams data/1-grams --ngrams data/9-grams
          --vtype binary --dims 100 --output coals-svdb-100.model

The assumed input format for the unigrams is:

    1|unigram_1|f
    1|unigram_2|f
    .|.........|.
    1|unigram_n|f

where *f* is an integer representing the frequency of the unigram. The assumed
input format for the n-grams, in turn, is:

    n|ngram_1|f
    n|ngram_2|f
    .|.......|...
    n|ngram_n|f

where *n* denotes the size of the *n*-gram, and *f* its frequency.

Dependencies
------------

COALS requires [uthash](http://troydhanson.github.io/uthash/) and
[svdlibc](https://github.com/lucasmaystre/svdlibc).

License
-------

COALS is available under the [Apache License, Version
2.0](http://www.apache.org/licenses/LICENSE-2.0.html).
