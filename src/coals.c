/*
 * coals.c
 *
 * Copyright 2012-2014 Harm Brouwer <me@hbrouwer.eu>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#include "../lib/uthash-2.0.1/src/uthash.h"
#include "../lib/SVDLIBC/svdlib.h"

#define VERSION "0.99-beta"

#define WTYPE_RAMPED 0
#define WTYPE_FLAT 1

#define VTYPE_REAL 0
#define VTYPE_BINARY 1
#define VTYPE_BINARY_PN 2

#define BUF_SIZE 131072

/*
 * This implements the Correlated Occurrence Analogue to Lexical Semantics
 * (COALS; Rohde, Gonnerman, & Plaut, 2005). COALS represents word meaning
 * by means of high-dimensional vectors derived from word co-occurrence
 * patterns in large corpora. The construction of these vectors starts by
 * compiling a co-occurrence table using a ramped (or alternatively, flat)
 * n-word window. Rohde et al. use a 4-word, ramped window:
 *
 *     1 2 3 4 [0] 4 3 2 1
 *
 * Next, all but the m (14.000 in case of Rohde et al.) columns, reflecting
 * the most frequent words, are discarded. Co-occurrence counts are then
 * converted to word-pair correlations; negative correlations set to zero,
 * and positive correlations are replaced by their square root in order to
 * reduce difference between them. The rows of the co-occurrence table then
 * represent COALS vectors for their respective words. Optionally, Singular
 * Value Decomposition (SVD) can be used to reduce the dimensionality of
 * these vectors, and these reduced vectors can, in turn, be converted to
 * binary vectors, by setting negative components to zero, and positive
 * components to one.
 *
 * The implementation also supports a recent extension to the COALS model
 * as proposed by Chang et al. (2012).
 *
 * References
 *
 * Chang, Y., Furber, S., & Welbourne, S. (2012). Generating Realistic
 *     Semantic Codes for Use in Neural Network Models. In Miyake, N.,
 *     Peebles, D, and Cooper, R. P. (Eds.). Proceedings of the 34th
 *     Annual Meeting of the Cognitive Science Society (CogSci 2012).
 *
 * Rohde, D. L. T., Gonnerman, L. M., & Plaut, D. C. (2005). An Improved
 *     Model of Semantic Similarity Based on Lexical Co-Occurrence.
 */

/*
 * ########################################################################
 * ## Data structures                                                    ##
 * ########################################################################
 */

/*
 * COALS configuration.
 */

struct config 
{
        int w_size;             /* window size */
        int w_type;             /* window type */
        int rows;               /* rows of the co-occurrence matrix */
        int cols;               /* columns of the co-occurrence matrix */
        int dims;               /* number of dimensions (for SVD) */
        int v_type;             /* vector type */
        char *unigrams_fn;      /* unigrams file name */
        char *ngrams_fn;        /* ngrams file name */
        char *output_fn;        /* output file name */
        char *enf_wds_fn;       /* enforced words file name */

        int positive_fts;       /* number of positive features */
        int negative_fts;       /* number of negative features */

        char *vectors_fn;       /* COALS vectors file name */
        int top_k;              /* number of similar words */
};

/*
 * Word and co-occurrence frequency hash table.
 */

struct freqs
{
        char *word;             /* word string (hash-key) */
        unsigned int freq;      /* word frequency */
        struct freqs *cfqs;     /* co-occurrence frequencies */
        UT_hash_handle hh;      /* hash handle */
};

/*
 * Hash table of words for which to enforce inclusion.
 */

struct enf_words
{
        char *word;             /* word string (hash-key) */
        UT_hash_handle hh;      /* hash handle */
};

/*
 * Hash table of COALS vectors.
 */
struct vectors
{
        char *word;             /* word string (hash-key) */
        double *vector;         /* COALS vectors */
        UT_hash_handle hh;      /* hash handle */
};

/*
 * ########################################################################
 * ## Function protoypes                                                 ##
 * ########################################################################
 */

bool config_is_sane(struct config *cfg);
void print_help(char *exec_name);
void print_version();

void coals(struct config *cfg);

void populate_freq_hash(struct config *cfg, struct freqs **fqs);
void populate_cfreq_hashes(struct config *cfg, struct freqs **fqs);
void dispose_freq_hash(struct freqs **fqs);
bool is_word(char *word);
int sort_by_freq(struct freqs *a, struct freqs *b);

void populate_enf_word_hash(struct config *cfg, struct freqs **fqs,
                struct enf_words **ewds);
void dispose_enf_word_hash(struct enf_words **ewds);

DMat construct_cm(struct config *cfg, struct freqs **fqs,
                struct enf_words **ewds);
void convert_freqs_to_correlations(DMat dcm);
double pearson_correlation(double val, double rt, double ct, double t);

DMat generate_reduced_vectors(struct config *cfg, DMat dcm, SVDRec svdrec);
DMat invert_singular_values(struct config *cfg, double *S_hat);
DMat multiply_matrices(DMat m1, DMat m2);

void write_vectors(struct config *cfg, struct freqs **fqs,
                struct enf_words **ewds, DMat cvs);
void fprint_real_vector(FILE *fd, struct config *cfg, char *w, DMat cvs,
                int r);
void fprint_binary_vector(FILE *fd, struct config *cfg, char *w, DMat cvs,
                int r);
void fprint_binary_pn_vector(FILE *fd, struct config *cfg, char *w, DMat cvs,
                int r);

void similarity(struct config *cfg);

void identify_model_parameters(struct config *cfg);
void populate_vector_hash(struct config *cfg, struct vectors **vecs);

DMat construct_sm(struct config *cfg, struct vectors **vecs);

double vector_mean(double *v, int dims);
double vector_similarity(double *v1, double v1_mean, double *v2,
                double v2_mean, int dims);

void write_topk_similar_words(struct config *cfg, struct vectors **vecs,
                DMat dsm);

/*
 * ########################################################################
 * ## Main functions                                                     ##
 * ########################################################################
 */

int main(int argc, char **argv)
{
        printf("\n");
        printf("COALS version %s\n", VERSION);
        printf("Copyright (c) 2012-2014 Harm Brouwer <me@hbrouwer.eu>\n");
        printf("Center for Language and Cognition, University of Groningen\n");
        printf("Netherlands Organisation for Scientific Research (NWO)\n");
        printf("\n");

        struct config *cfg;
        if (!(cfg = malloc(sizeof(struct config))))
                goto error_out;
        memset(cfg, 0, sizeof(struct config));

        /* read options */
        for (int i = 0; i < argc; i++) {
                /* window size */
                if (strcmp(argv[i], "--wsize") == 0) {
                        if (++i < argc)
                                cfg->w_size = atoi(argv[i]);
                }

                /* window type */
                if (strcmp(argv[i], "--wtype") == 0) {
                        if (++i < argc) {
                                if (strcmp(argv[i], "ramped") == 0)
                                        cfg->w_type = WTYPE_RAMPED;
                                if (strcmp(argv[i], "flat") == 0)
                                        cfg->w_type = WTYPE_FLAT;
                        }
                }

                /* rows */
                if (strcmp(argv[i], "--rows") == 0) {
                        if (++i < argc)
                                cfg->rows = atoi(argv[i]);
                }

                /* columns */
                if (strcmp(argv[i], "--cols") == 0) {
                        if (++i < argc)
                                cfg->cols = atoi(argv[i]);
                }

                /* dimensions */
                if (strcmp(argv[i], "--dims") == 0) {
                        if (++i < argc)
                                cfg->dims = atoi(argv[i]);
                }

                /* vector type */
                if (strcmp(argv[i], "--vtype") == 0) {
                        if (++i < argc) {
                                if (strcmp(argv[i], "real") == 0)
                                        cfg->v_type = VTYPE_REAL;
                                if (strcmp(argv[i], "binary") == 0)
                                        cfg->v_type = VTYPE_BINARY;
                                if (strcmp(argv[i], "binary_pn") == 0)
                                        cfg->v_type = VTYPE_BINARY_PN;
                        }
                }
                
                /* unigrams */
                if (strcmp(argv[i], "--unigrams") == 0) {
                        if (++i < argc)
                                cfg->unigrams_fn = argv[i];
                }

                /* ngrams */
                if (strcmp(argv[i], "--ngrams") == 0) {
                        if (++i < argc)
                                cfg->ngrams_fn = argv[i];
                }

                /* output */
                if (strcmp(argv[i], "--output") == 0) {
                        if (++i < argc)
                                cfg->output_fn = argv[i];
                }

                /* output */
                if (strcmp(argv[i], "--enforce") == 0) {
                        if (++i < argc)
                                cfg->enf_wds_fn = argv[i];
                }

                /* positive features */
                if (strcmp(argv[i], "--pos_fts") == 0) {
                        if (++i < argc)
                                cfg->positive_fts = atoi(argv[i]);
                }

                /* negative features */
                if (strcmp(argv[i], "--neg_fts") == 0) {
                        if (++i < argc)
                                cfg->negative_fts = atoi(argv[i]);
                }

                /* vectors */
                if (strcmp(argv[i], "--vectors") == 0) {
                        if (++i < argc)
                                cfg->vectors_fn = argv[i];
                }

                /* top-k similar words */
                if (strcmp(argv[i], "--topk") == 0) {
                        if (++i < argc)
                                cfg->top_k = atoi(argv[i]);
                }

                /* help */
                if (strcmp(argv[i], "--help") == 0) {
                        print_help(argv[0]);
                        exit(EXIT_SUCCESS);
                }

                /* version */
                if (strcmp(argv[i], "--version") == 0) {
                        print_version();
                        exit(EXIT_SUCCESS);
                }
        }

        if (!config_is_sane(cfg)) {
                print_help(argv[0]);
                exit(EXIT_FAILURE);
        }

        if (!cfg->vectors_fn)
                coals(cfg);
        else
                similarity(cfg);

        free(cfg);

        exit(EXIT_SUCCESS);

error_out:
        perror("[main()]");
        exit(EXIT_FAILURE);
}

/*
 * Checks the sanity of a COALS configuration.
 */

bool config_is_sane(struct config *cfg)
{
        if (cfg->output_fn == NULL)
                return false;

        if (!cfg->vectors_fn) {
                if (cfg->w_size == 0)
                        return false;
                if (cfg->rows == 0)
                        return false;
                if (cfg->cols == 0)
                        return false;
                if (cfg->unigrams_fn == NULL)
                        return false;
                if (cfg->ngrams_fn == NULL)
                        return false;

                if (cfg->v_type == VTYPE_BINARY_PN) {
                        if (cfg->positive_fts == 0 || cfg->positive_fts > cfg->dims)
                                return false;
                        if (cfg->negative_fts == 0 || cfg->negative_fts > cfg->dims)
                                return false;
                }
        } else {
                if (cfg->top_k == 0)
                        return false;
        }

        return true;
}

/*
 * Print help.
 */

void print_help(char *exec_name)
{
        printf(
                        "usage %s [options]\n\n"

                        "  constructing COALS vectors:\n"
                        "    --wsize <num>\tset co-occurrence window size to <num>\n"
                        "    --wtype <type>\tuse <type> window for co-occurrences (dflt: ramped)\n"
                        "    --rows <num>\tset number of rows of the co-occurrence matrix to <num>\n"
                        "    --cols <num>\tset number of cols of the co-occurrence matrix to <num>\n"
                        "    --dims <num>\treduce COALS vectors to <num> dimensions (using SVD)\n"
                        "    --vtype <type>\tconstruct <type> (real/binary[_pn]) vectors (dflt: real)\n"
                        "    --unigrams <file>\tread unigram counts (word frequencies) from <file>\n"
                        "    --ngrams <file>\tread n-gram counts (co-occurrence freqs.) from <file>\n"
                        "    --output <file>\twrite COALS vectors to <file>\n"
                        "    --enforce <file>\tenforce inclusion of words in <file>\n"
                        
                        "\n"
                        "    --pos_fts <num>\tnumber of positive features (for binary_pn vectors)\n"
                        "    --neg_fts <num>\tnumber of negative features (for binary_pn vectors)\n"

                        "\n"
                        "  extracting similar words:\n"
                        "    --vectors <file>\tcompute similarities on basis of vectors in <file>\n"
                        "    --output <file>\twrite top-k similar word sets to <file>\n"
                        "    --topk <num>\textract top-<num> similar words for each word\n"

                        "\n"
                        "  basic information for users:\n"
                        "    --help\t\tshows this help message\n"
                        "    --version\t\tshows version\n"
                        "\n",
                        exec_name);
}

void print_version()
{
        printf("%s\n", VERSION);
}

/*
 * Main COALS function.
 */

void coals(struct config *cfg)
{
        fprintf(stderr, "--- starting construction of COALS vectors (...)\n");

        fprintf(stderr, "\tco-occurrence window size:\t[%d]\n", cfg->w_size);
        fprintf(stderr, "\tco-occurrence window type:\t");
        if (cfg->w_type == WTYPE_RAMPED)
                fprintf(stderr, "[ramped]\n");
        if (cfg->w_type == WTYPE_FLAT)
                fprintf(stderr, "[flat]\n");

        fprintf(stderr, "\tco-occurrence matrix rows:\t[%d]\n", cfg->rows);
        fprintf(stderr, "\tco-occurrence matrix cols:\t[%d]\n", cfg->cols);

        /*
         * If no dimension reduction is requested, vector dimensionality
         * equals the number of columns of the co-occurrence matrix.
         */
        if (cfg->dims == 0)
                cfg->dims = cfg->cols;

        fprintf(stderr, "\tCOALS vector dimensions:\t[%d]\n", cfg->dims);
        fprintf(stderr, "\tCOALS vector type:\t\t");
        if (cfg->v_type == VTYPE_REAL)
                fprintf(stderr, "[real]\n");
        if (cfg->v_type == VTYPE_BINARY)
                fprintf(stderr, "[binary]\n");
        if (cfg->v_type == VTYPE_BINARY_PN)
                fprintf(stderr, "[binary_pn]\n");

        if (cfg->v_type == VTYPE_BINARY_PN) {
                fprintf(stderr, "\tnumber of positive features\t[%d]\n",
                                cfg->positive_fts);
                fprintf(stderr, "\tnumber of positive features\t[%d]\n",
                                cfg->negative_fts);
        }

        fprintf(stderr, "\tunigrams file:\t\t\t[%s]\n", cfg->unigrams_fn);
        fprintf(stderr, "\tn-grams file:\t\t\t[%s]\n", cfg->ngrams_fn);
        fprintf(stderr, "\toutput file:\t\t\t[%s]\n", cfg->output_fn);
        if (cfg->enf_wds_fn)
                fprintf(stderr, "\tinclude words from file:\t[%s]\n",
                                cfg->enf_wds_fn);

        /*
         * Populate frequency hash.
         */
        struct freqs *fqs = NULL;
        fprintf(stderr, "--- populating frequency hash from: [%s] (...)\n",
                        cfg->unigrams_fn);
        populate_freq_hash(cfg, &fqs);
        fprintf(stderr, "\twords read:\t\t\t[%d]\n", HASH_COUNT(fqs));

        /*
         * Populate co-occurrence frequency hashes.
         */
        fprintf(stderr, "--- populating co-occurrence frequency hashes from: [%s] (...)\n",
                        cfg->ngrams_fn);
        populate_cfreq_hashes(cfg, &fqs);

        /*
         * Populate hash of words that should be included
         * in the co-occurrence matrix, even though they do
         * not occur in the top-k most frequent words (only
         * if required).
         */
        struct enf_words *ewds = NULL;
        if (cfg->enf_wds_fn) {
                fprintf(stderr, "--- populating enforced word hash from: [%s] (...)\n",
                        cfg->enf_wds_fn);
                populate_enf_word_hash(cfg, &fqs, &ewds);
                fprintf(stderr, "\twords read:\t\t\t[%d]\n", HASH_COUNT(ewds));
        }

        /*
         * Construct a co-occurrence matrix on basis of
         * the co-occurrence frequency hashes.
         */
        fprintf(stderr, "--- constructing a co-occurrence matrix (...)\n");
        DMat dcm = construct_cm(cfg, &fqs, &ewds);
        fprintf(stderr, "\tbuilt matrix:\t\t\t[%ldx%ld]\n", dcm->rows, dcm->cols);

        /*
         * Convert frequencies to correlations.
         */
        fprintf(stderr, "--- converting frequencies to correlation coefficients (...)\n");
        convert_freqs_to_correlations(dcm);

        /*
         * Reduce dimensionality with SVD (if required).
         */
        SMat scm = NULL; SVDRec svdrec = NULL;
        if (cfg->dims > 0 && cfg->dims < cfg->cols) {
                fprintf(stderr, "--- reducing dimensionality with singular value decomposition (...)\n");
                scm = svdConvertDtoS(dcm);
                svdrec = svdLAS2A(scm, cfg->dims);
        }

        /*
         * Extract vectors. There are two options:
         *
         * 1) If the dimensionality of the vectors was reduced using
         *    SVD, we compute the matrix:
         *
         *    X * V^ * S^-1
         *
         *    in which each row represents a k-dimensional word vector.
         *
         * 2) If the dimensionality of the vectors was not reduced,
         *    the vectors are simply the COALS vectors computed in
         *    the frequency to correlation coefficients conversion 
         *    step. 
         */
        fprintf(stderr, "--- extracting COALS vectors (...)\n");
        DMat cvs = NULL;
        /* option 1 */
        if (svdrec) {
                cvs = generate_reduced_vectors(cfg, dcm, svdrec);
                svdFreeDMat(dcm);
                svdFreeSVDRec(svdrec);
        /* option 2 */
        } else {
                cvs = dcm;
        }

        /*
         * Write COALS vectors to output file.
         */
        fprintf(stderr, "--- writing coals vectors to: [%s] (...)\n",
                        cfg->output_fn);
        write_vectors(cfg, &fqs, &ewds, cvs);

        /* clean up */
        fprintf(stderr, "--- cleaning up (...)\n");
        svdFreeDMat(cvs);
        dispose_enf_word_hash(&ewds);
        dispose_freq_hash(&fqs);
}

/*
 * Populate word frequencies on basis of unigram counts.
 *
 * This function assumes the following file format:
 *
 * 1|<unigram_1>|<frequency>
 * 1|<unigram_2>|<frequency>
 * .|...........|...........
 *
 * where the first integer denotes that the frequency is
 * a unigram count.
 */

void populate_freq_hash(struct config *cfg, struct freqs **fqs)
{
        FILE *fd;
        if (!(fd = fopen(cfg->unigrams_fn, "r")))
                goto error_out;

        /* read all unigrams */
        char buf[BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                char *wp = index(buf, '|');
                char *fp = rindex(buf, '|');

                /* isolate word */
                char *word;
                int len = fp - ++wp;
                int block_size = (len + 1) * sizeof(char);
                if (!(word = malloc(block_size)))
                        goto error_out;
                memset(word, 0, block_size);
                strncpy(word, wp, len);

                /* isolate word frequency */
                unsigned int freq; // = atoi(++fp);
                sscanf(++fp, "%u", &freq);

                /* convert word to uppercase */
                for (int i = 0; i < strlen(word); i++)
                        word[i] = toupper(word[i]);

                /* skip if not a word */
                if (!is_word(word)) {
                        free(word);
                        continue;
                }

                /* 
                 * Add word to frequency hash. There are two options:
                 *
                 * 1) We are dealing with a new word, so we add a new
                 *    hash entry.
                 *
                 * 2) We have already seen this word, and we simply 
                 *    update its frequency.
                 */ 
                struct freqs *f;
                HASH_FIND_STR(*fqs, word, f);
                /* option 1 */
                if (!f) {
                        struct freqs *nf;
                        if (!(nf = malloc(sizeof(struct freqs))))
                                goto error_out;
                        memset(nf, 0, sizeof(struct freqs));
                        nf->word = word;
                        nf->freq = freq;
                        HASH_ADD_KEYPTR(hh, *fqs, nf->word, strlen(nf->word), nf);
                /* option 2 */
                } else {
                        f->freq += freq;
                        free(word);
                }
        }

        fclose(fd);

        /* sort hash on frequency (descending) */
        HASH_SORT(*fqs, sort_by_freq);

        return;

error_out:
        perror("[populate_freq_hash()]");
        return;
}

/*
 * Populate co-occurrence frequency hashes on basis of n-gram counts.
 *
 * This function assumes the following file format:
 *
 * n|<ngram_1>|<frequency>
 * n|<ngram_2>|<frequency>
 * .|.........|...........
 *
 * where the first integer denotes that the frequency is
 * a n-gram count for n-grams of size n.
 */

void populate_cfreq_hashes(struct config *cfg, struct freqs **fqs)
{
        FILE *fd;
        if (!(fd = fopen(cfg->ngrams_fn, "r")))
                goto error_out;

        /* n-gram size and word index */
        int ngram_sz = cfg->w_size * 2 + 1;
        int word_idx = cfg->w_size;

        /* read all n-grams */
        char buf[BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                char *wp = index(buf, '|');
                char *fp = rindex(buf, '|');

                /* isolate n-gram */
                char *ngram;
                int len = fp - ++wp;
                int block_size = (len + 1) * sizeof(char);
                if (!(ngram = malloc(block_size)))
                        goto error_out;
                memset(ngram, 0, block_size);
                strncpy(ngram, wp, len);

                /* isolate n-gram frequency */
                unsigned int freq; //atoi(++fp);
                sscanf(++fp, "%u", &freq);

                /* convert n-gram to uppercase */
                for (int i = 0; i < strlen(ngram); i++)
                        ngram[i] = toupper(ngram[i]);

                /* isolate individual words */
                char *words[ngram_sz];
                words[0] = strtok(ngram, " ");
                for (int i = 1; i < ngram_sz; i++)
                        words[i] = strtok(NULL, " ");

                /* skip if string in focus is not a word */
                if (!is_word(words[word_idx])) {
                        free(ngram);
                        continue;
                }

                /* 
                 * skip if there is no frequency for
                 * word in focus
                 */
                struct freqs *f;
                HASH_FIND_STR(*fqs, words[word_idx], f);
                if (!f) {
                        free(ngram);
                        continue;
                }

                /* 
                 * add co-occurence frequencies for all words
                 * to the left and right of the focused word
                 */
                for (int i = 0; i < ngram_sz; i++) {
                        /* skip word in focus */
                        if (i == word_idx)
                                continue;

                        /* skip if not a word */
                        if (!is_word(words[i]))
                                continue;

                        /* frequency */
                        unsigned int rfreq = freq;

                        /* ramp frequency (if required) */
                        if (cfg->w_type == WTYPE_RAMPED) {
                                if (i < word_idx)
                                        rfreq *= i + 1;
                                if (i > word_idx)
                                        rfreq *= ngram_sz - i;
                        }

                        /* 
                         * Add word to co-occurrence frequency hash.
                         * There are two options:
                         *
                         * 1) We are dealing with a new word, so we add
                         *    a new hash entry.
                         *
                         * 2) We have already seen this word, and we
                         *    simply update its co-occurrence frequency.
                         */
                        struct freqs *cf;
                        HASH_FIND_STR(f->cfqs, words[i], cf);
                        /* option 1 */
                        if (!cf) {
                                struct freqs *nf;
                                if (!(nf = malloc(sizeof(struct freqs))))
                                        goto error_out;
                                memset(nf, 0, sizeof(struct freqs));
                                block_size = (strlen(words[i]) + 1) * sizeof(char);
                                if (!(nf->word = malloc(block_size)))
                                        goto error_out;
                                memset(nf->word, 0, block_size);
                                strncpy(nf->word, words[i], strlen(words[i]));
                                nf->freq = rfreq;
                                HASH_ADD_KEYPTR(hh, f->cfqs, nf->word, strlen(nf->word), nf);
                        /* option 2 */
                        } else {
                                cf->freq += rfreq;
                        }
                }
                free(ngram);
        }

        fclose(fd);

        return;

error_out:
        perror("[populate_cfreq_hashes()]");
        return;
}

/*
 * Dispose frequency hash table (and recursively its
 * embedded co-occurrence frequency tables).
 */

void dispose_freq_hash(struct freqs **fqs)
{
        struct freqs *f, *nf;
        HASH_ITER(hh, *fqs, f, nf) {
                if (f->cfqs)
                        dispose_freq_hash(&f->cfqs);
                HASH_DEL(*fqs, f);
                free(f);
        }
}

/*
 * Checks whether a string is a word.
 */

bool is_word(char *word)
{
        /*
         * String is a non-word if it contains
         * punctuation characthers other than a
         * dash.
         */
        for (int i = 0; i < strlen(word); i++)
                if (!isalpha(word[i]) && word[i] != '-')
                        return false;

        /*
         * String is a non-word if its only
         * character is a punctuation character.
         */
        if (strlen(word) == 1)
                if (ispunct(word[0]))
                        return false;

        return true;
}

/*
 * Sort hash on basis of frequency (descending).
 */

int sort_by_freq(struct freqs *a, struct freqs *b)
{

        if (a->freq < b->freq)
                return 1;
        else if (a->freq == b->freq)
                return 0;
        else
                return -1;
}

/*
 * Populate hash of enforced words.
 *
 * This function assumes the following file format:
 *
 * <word_1>
 * <word_2>
 * ........
 */

void populate_enf_word_hash(struct config *cfg, struct freqs **fqs,
                struct enf_words **ewds)
{
        FILE *fd;
        if (!(fd = fopen(cfg->enf_wds_fn, "r")))
                goto error_out;

        /* read all words */
        char buf[BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                /* copy word */
                char *word;
                int block_size = strlen(buf) * sizeof(char);
                if (!(word = malloc(block_size)))
                        goto error_out;
                memset(word, 0, block_size);
                strncpy(word, buf, strlen(buf) - 1); 
                
                /*
                 * We only want to add a word if it occurs in the corpus,
                 * which is the case if we have its frequency.
                 */
                struct freqs *f;
                HASH_FIND_STR(*fqs, word, f);
                if (f) {
                        struct enf_words *ew;
                        if (!(ew = malloc(sizeof(struct enf_words))))
                                goto error_out;
                        memset(ew, 0, sizeof(struct enf_words));
                        ew->word = word;
                        HASH_ADD_KEYPTR(hh, *ewds, ew->word, strlen(ew->word), ew);
                } else {
                        fprintf(stderr, "\tskipping unknown word:\t\t(%s)\n", word);
                        free(word);
                }
        }

        /* 
         * Delete words that occur in the top-k frequencies, as
         * we do not want the same word to occur twice.
         */
        int r; struct freqs *f; 
        for (r = 0, f = *fqs; r < cfg->rows && f != NULL; r++, f = f->hh.next) {
                struct enf_words *ew;
                HASH_FIND_STR(*ewds, f->word, ew);
                if (ew) {
                        HASH_DEL(*ewds, ew);
                }
        }

        fclose(fd);

        return;

error_out:
        perror("[populate_enf_word_hash()]");
        return;
}

/*
 * Dispose enforced words hash.
 */

void dispose_enf_word_hash(struct enf_words **ewds)
{
        struct enf_words *ew, *new;
        HASH_ITER(hh, *ewds, ew, new) {
                HASH_DEL(*ewds, ew);
                free(ew);
        }
}

/*
 * Construct a co-occurrence matrix.
 */

DMat construct_cm(struct config *cfg, struct freqs **fqs,
                struct enf_words **ewds)
{
        int extra_rows = HASH_COUNT(*ewds);
        DMat dcm = svdNewDMat(cfg->rows + extra_rows, cfg->cols);

        /* enter co-occurrence frequencies for the top-k words */
        int r; struct freqs *rf;
        for (r = 0, rf = *fqs; r < (dcm->rows - extra_rows) && rf != NULL; r++, rf = rf->hh.next) {
                int c; struct freqs *cf;
                for (c = 0, cf = *fqs; c < dcm->cols && cf != NULL; c++, cf = cf->hh.next) {
                        /*
                         * If there is a co-occurence frequency for the
                         * current word pair, enter its value into the
                         * matrix. Otherwise, enter zero.
                         */
                        struct freqs *f;
                        HASH_FIND_STR(rf->cfqs, cf->word, f);
                        if (f)
                                dcm->value[r][c] = f->freq;
                        else
                                dcm->value[r][c] = 0.0;
                }
        }

        /* enter co-occurrence frequencies for enforced words */
        struct enf_words *ew;
        for (r = (dcm->rows - extra_rows), ew = *ewds; r < dcm->rows && ew != NULL; r++, ew = ew->hh.next) {
                int c; struct freqs *cf;
                for (c = 0, cf = *fqs; c < dcm->cols && cf != NULL; c++, cf = cf->hh.next) {
                        /*
                         * If there is a co-occurence frequency for the
                         * current word pair, enter its value into the
                         * matrix. Otherwise, enter zero.
                         */
                        HASH_FIND_STR(*fqs, ew->word, rf);
                        struct freqs *f;
                        HASH_FIND_STR(rf->cfqs, cf->word, f);
                        if (f)
                                dcm->value[r][c] = f->freq;
                        else
                                dcm->value[r][c] = 0.0;
                }
        }

        return dcm;
}

/*
 * Convert raw frequencies to correlation coefficients. Negative
 * correlations are set to zero, and positive correlations are replaced by
 * their square root.
 */

void convert_freqs_to_correlations(DMat dcm)
{
        DMat rts = svdNewDMat(dcm->rows, 1);
        DMat cts = svdNewDMat(1, dcm->cols);
        double t = 0.0;

        /* determine row, column, and grand total */
        for (int r = 0; r < dcm->rows; r++) {
                for (int c = 0; c < dcm->cols; c++) {
                        rts->value[r][0] += dcm->value[r][c];
                        cts->value[0][c] += dcm->value[r][c];
                        t += dcm->value[r][c];
                }
        }

        /*
         * Replace frequencies with correlations. If
         * a correlation is negative, set its corresponding
         * cell to zero. If it is positive, by contrast,
         * take its square root.
         */
        for (int r = 0; r < dcm->rows; r++) {
                for (int c = 0; c < dcm->cols; c++) {
                        double pc = pearson_correlation(dcm->value[r][c],
                                        rts->value[r][0], cts->value[0][c], t);
                        if (pc < 0)
                                dcm->value[r][c] = 0.0;
                        else
                                dcm->value[r][c] = sqrt(pc);
                }
        }

        svdFreeDMat(rts);
        svdFreeDMat(cts);
}

/*
 * Pearson's correlation coefficient:
 *
 *           T * w_a,b - sum_j w_a,j * sum_i w_i,b
 * pc_a,b = ---------------------------------------
 *          (sum_j w_a,j * (T - sum_j w_a,j) * 
 *           sum_i w_i,b * (T - sum_i w_i,b)) ^ 0.5
 *
 * where: 
 *  
 * T = sum_i sum_j w_i,j
 */

double pearson_correlation(double val, double rt, double ct, double t)
{
        double nom = t * val - rt * ct;
        double denom = pow(rt * (t - rt) * ct * (t - ct), 0.5);

        if (denom == 0.0) {
                return 0.0;
        } else {
                return nom / denom;
        }
}

/*
 * Generate a matrix of COALS vectors with reduced dimensionality.
 * This is done by computing:
 * 
 * X * V^ * S^-1
 *
 * where X is the matrix of COALS vectors as computed in the frequency
 * to correlation coefficients conversion step.
 */

DMat generate_reduced_vectors(struct config *cfg, DMat dcm, SVDRec svdrec)
{
        /* 
         * SVD returns V^t. Transpose V^t to get V^.
         */
        fprintf(stderr, "\ttransposing V^t (...)\n");
        DMat V_hat = svdTransposeD(svdrec->Vt);

        /*
         * SVD returns S^. Invert S^ to get S^-1.
         */
        fprintf(stderr, "\tinverting S^ (...)\n");
        DMat S_hat_inv = invert_singular_values(cfg, svdrec->S);

        /*
         * Compute V^ * S^-1.
         */
        fprintf(stderr, "\tcomputing V^ * S^-1 (...)\n");
        DMat V_hat_x_S_hat_inv = multiply_matrices(V_hat, S_hat_inv);

        /* free V^ and S^-1 */
        svdFreeDMat(V_hat);
        svdFreeDMat(S_hat_inv);

        /*
         * Compute X * (V^ * S^-1).
         */
        fprintf(stderr, "\tcomputing X * (V^ * S^-1) (...)\n");
        DMat cvs = multiply_matrices(dcm, V_hat_x_S_hat_inv);

        /* free V^ * S^-1 */
        free(V_hat_x_S_hat_inv);

        return cvs;
}

/*
 * Invert singular values. This function takes an array of singular values,
 * which correspond to the diagonal cells of the S^ matrix that results 
 * from SVD, and returns a matrix S^-1 that is the inverse of S^:
 *
 *      [ s_1  0   0   ]              [ (1/s_1)    0       0    ]
 * S^ = [  0  ...  0   ]       S^-1 = [    0    .......    0    ]
 *      [  0   0  s_n  ]              [    0       0    (1/s_n) ]
 * 
 * WARNING: This function only works properly for matrices that only have
 * non-zero values in their diagonal cells (like the identity matrix).
 */

DMat invert_singular_values(struct config *cfg, double *S_hat)
{
        DMat S_hat_inv = svdNewDMat(cfg->dims, cfg->dims);

        for (int r = 0; r < S_hat_inv->rows; r++)
                for (int c = 0; c < S_hat_inv->cols; c++)
                        if (r == c)
                                S_hat_inv->value[r][c] = 1.0 / S_hat[r];
                        else
                                S_hat_inv->value[r][c] = 0.0;

        return S_hat_inv;
}

/*
 * Multiply two matrices:
 *
 * (n x m) * (m x p) = (n x p)
 *
 *             [ g h ]
 * [ a b c ]   [ i j ]   [ (a * g + b * i + c * k) (a * h + b * j + c * l) ]
 * [ d e f ] x [ k l ] = [ (d * g + e * i + f * k) (d * h + e * j + f * l) ]
 */

DMat multiply_matrices(DMat m1, DMat m2)
{
        DMat m3 = svdNewDMat(m1->rows, m2->cols);

        for (int r = 0; r < m3->rows; r++)
                for (int c = 0; c < m3->cols; c++)
                        for (int i = 0; i < m1->cols; i++)
                                m3->value[r][c] += m1->value[r][i] * m2->value[i][c];

        return m3;
}

/*
 * Write COALS vectors to output file.
 */

void write_vectors(struct config *cfg, struct freqs **fqs,
                struct enf_words **ewds, DMat cvs)
{
        FILE *fd;
        if (!(fd = fopen(cfg->output_fn, "w")))
                goto error_out;

        /* determine extra rows (for enforced words) */
        int extra_rows = HASH_COUNT(*ewds);

        int num_written = 0;

        /* write vectors of the top-k words */
        int r; struct freqs *f;
        for (r = 0, f = *fqs; r < (cvs->rows - extra_rows) && f != NULL; r++, f = f->hh.next) {
                if (f->cfqs == NULL) {
                        fprintf(stderr, "\tskipping invalid vector:\t(%s)\n", f->word);
                        continue;
                }

                if (cfg->v_type == VTYPE_REAL)
                        fprint_real_vector(fd, cfg, f->word, cvs, r);
                if (cfg->v_type == VTYPE_BINARY)
                        fprint_binary_vector(fd, cfg, f->word, cvs, r);
                if (cfg->v_type == VTYPE_BINARY_PN)
                        fprint_binary_pn_vector(fd, cfg, f->word, cvs, r);

                num_written++;
        }

        /* enter vectors of enforced words */
        struct enf_words *ew;
        for (r = (cvs->rows - extra_rows), ew = *ewds; r < cvs->rows && ew != NULL; r++, ew = ew->hh.next) {
                HASH_FIND_STR(*fqs, ew->word, f);
                if (f->cfqs == NULL) {
                        fprintf(stderr, "\tskipping invalid vector:\t(%s)\n", ew->word);
                        continue;
                }

                if (cfg->v_type == VTYPE_REAL)
                        fprint_real_vector(fd, cfg, ew->word, cvs, r);
                if (cfg->v_type == VTYPE_BINARY)
                        fprint_binary_vector(fd, cfg, ew->word, cvs, r);
                if (cfg->v_type == VTYPE_BINARY_PN)
                        fprint_binary_pn_vector(fd, cfg, ew->word, cvs, r);
                        
                num_written++;
        }

        fclose(fd);

        fprintf(stderr, "\tvectors written:\t\t[%d]\n", num_written);

        return;

error_out:
        perror("[write_vectors()]");
        return;
}

/*
 * Print a real vector to a file.
 */

void fprint_real_vector(FILE *fd, struct config *cfg, char *w, DMat cvs,
                int r)
{
        fprintf(fd, "\"%s\"", w);
        for (int c = 0; c < cvs->cols; c++) {
                fprintf(fd, ",%f", cvs->value[r][c]);
        }
        fprintf(fd, "\n");
}

/*
 * Print a binary vector to a file.
 */

void fprint_binary_vector(FILE *fd, struct config *cfg, char *w, DMat cvs,
                int r)
{
        fprintf(fd, "\"%s\"", w);
        for (int c = 0; c < cvs->cols; c++) {
                if (cvs->value[r][c] > 0) {
                        fprintf(fd, ",1");
                } else {
                        fprintf(fd, ",0");
                }
        }
        fprintf(fd, "\n");
}

/*
 * Print a binary_pn vector to a file.
 *
 * This implements an extension to COALS that incorporates both a fixed
 * number of positive and negative features into vectors (Chang, Furber, &
 * Welbourne, 2012). The main idea is to concatenate two k-dimensional
 * vectors, producing a 2k-dimensional vector, and to set the the n-most
 * positive features to 1 in the first k units, and to set the m-most
 * negative features to 1 in the second k units.
 *
 * [ u_1 ... u_k ] --> [ u_1 ... u_k, u_n+1 ... u_2k ]
 *                       |_________|  |____________|
 *                            |             |
 *                        positive       negative
 *                        features       features
 *
 * References
 *
 * Chang, Y., Furber, S., & Welbourne, S. (2012). Generating Realistic
 *     Semantic Codes for Use in Neural Network Models. In Miyake, N.,
 *     Peebles, D, and Cooper, R. P. (Eds.). Proceedings of the 34th
 *     Annual Meeting of the Cognitive Science Society (CogSci 2012).
 */

void fprint_binary_pn_vector(FILE *fd, struct config *cfg, char *w, DMat cvs,
                int r)
{
        int pos_fts_idxs[cfg->positive_fts];
        int neg_fts_idxs[cfg->negative_fts];

        /* identify the n most positive features */
        for (int i = 0; i < cfg->positive_fts; i++) {
                pos_fts_idxs[i] = -1;
                for (int c = 0; c < cvs->cols; c++) {
                        /* skip if we already have this feature */
                        bool seen = false;
                        for (int j = 0; j < i; j++)
                                if (pos_fts_idxs[j] == c)
                                        seen = true;
                        if (seen)
                                continue;
                        if (pos_fts_idxs[i] == -1)
                                pos_fts_idxs[i] = c;
                        else if (cvs->value[r][c] > cvs->value[r][pos_fts_idxs[i]]
                                        && cvs->value[r][c] > 0.0)
                                pos_fts_idxs[i] = c;
                }
        }

        /* identify the m most negative features */
        for (int i = 0; i < cfg->negative_fts; i++) {
                neg_fts_idxs[i] = -1;
                for (int c = 0; c < cvs->cols; c++) {
                        /* skip if we already have this feature */
                        bool seen = false;
                        for (int j = 0; j < i; j++)
                                if (neg_fts_idxs[j] == c)
                                        seen = true;
                        if (seen)
                                continue;
                        /* skip if feature is a positive feature */
                        bool is_pos = false;
                        for (int j = 0; j < cfg->positive_fts; j++)
                                if (pos_fts_idxs[j] == c)
                                        is_pos = true;
                        if (is_pos)
                                continue;
                        if (neg_fts_idxs[i] == -1)
                                neg_fts_idxs[i] = c;
                        else if (cvs->value[r][c] < cvs->value[r][neg_fts_idxs[i]]
                                        && cvs->value[r][c] < 0.0)
                                neg_fts_idxs[i] = c;
                }
        }

        fprintf(fd, "\"%s\"", w);

        /* write half with positive features active */
        for (int c = 0; c < cvs->cols; c++) {
                bool on = false;
                for (int i = 0; i < cfg->positive_fts; i++)
                        if (pos_fts_idxs[i] == c)
                                on = true;
                if (on)
                        fprintf(fd, ",1");
                else
                        fprintf(fd, ",0");
        }

        /* write half with negative features active */
        for (int c = 0; c < cvs->cols; c++) {
                bool on = false;
                for (int i = 0; i < cfg->negative_fts; i++)
                        if (neg_fts_idxs[i] == c)
                                on = true;
                if (on)
                        fprintf(fd, ",1");
                else
                        fprintf(fd, ",0");
        }

        fprintf(fd, "\n");
}

/*
 * ########################################################################
 * ## Extract similar words                                              ##
 * ########################################################################
 */

/*
 * Main similarity function.
 */

void similarity(struct config *cfg)
{
        fprintf(stderr, "--- starting extraction of top-k similar words (...)\n");

        fprintf(stderr, "\tvectors file:\t\t\t[%s]\n", cfg->vectors_fn);
        fprintf(stderr, "\toutput file:\t\t\t[%s]\n", cfg->output_fn);
        fprintf(stderr, "\ttop-k:\t\t\t\t[%d]\n", cfg->top_k);

        fprintf(stderr, "--- identifying model parameters from vectors file: [%s] (...)\n",
                        cfg->vectors_fn);
        identify_model_parameters(cfg);
        fprintf(stderr, "\tnumber of vectors:\t\t[%d]\n", cfg->rows);
        fprintf(stderr, "\tvector dimensionality:\t\t[%d]\n", cfg->dims);

        fprintf(stderr, "--- populating vectors hash from: [%s] (...)\n",
                        cfg->vectors_fn);
        struct vectors *vecs = NULL;
        populate_vector_hash(cfg, &vecs);
        fprintf(stderr, "\tvectors read:\t\t\t[%d]\n", HASH_COUNT(vecs));

        fprintf(stderr, "--- constructing a similarity matrix (...)\n");
        DMat dsm = construct_sm(cfg, &vecs);
        fprintf(stderr, "\tbuilt matrix:\t\t\t[%ldx%ld]\n", dsm->rows, dsm->cols);

        fprintf(stderr, "--- writing top-k most similar words to: [%s] (...)\n",
                        cfg->output_fn);
        write_topk_similar_words(cfg, &vecs, dsm);

        fprintf(stderr, "--- cleaning up (...)\n");
        svdFreeDMat(dsm);
}

void identify_model_parameters(struct config *cfg)
{
        FILE *fd;
        if (!(fd = fopen(cfg->vectors_fn, "r")))
                goto error_out;

        char buf[BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                /* determine number of columns */
                if (cfg->cols == 0) {
                        strtok(buf, ",");
                        while (strtok(NULL, ","))
                                cfg->cols++;
                }
                /* count rows */
                cfg->rows++;
        }

        cfg->dims = cfg->cols;

        fclose(fd);

        return;

error_out:
        perror("[identify_model_parameters()]");
        return;
}

/*
 * Populate vector hash from file.
 */

void populate_vector_hash(struct config *cfg, struct vectors **vecs)
{
        FILE *fd;
        if (!(fd = fopen(cfg->vectors_fn, "r")))
                goto error_out;

        /* read all vectors */
        char buf[BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                struct vectors *v;
                if (!(v = malloc(sizeof(struct vectors))))
                        goto error_out;
                memset(v, 0, sizeof(struct vectors));
                
                char *fq = index(buf, '"');
                char *lq = rindex(buf, '"');
               
                /* isolate word */
                int len = lq - ++fq;
                int block_size = (len + 1) * sizeof(char);
                if (!(v->word = malloc(block_size)))
                        goto error_out;
                memset(v->word, 0, block_size);
                strncpy(v->word, fq, len);

                /* isolate vector */
                block_size = cfg->cols * sizeof(double);
                if (!(v->vector = malloc(block_size)))
                        goto error_out;
                memset(v->vector, 0, block_size);
                char *tok = strtok(buf, ",");
                for (int i = 0; i < cfg->cols; i++) {
                        tok = strtok(NULL, ",");
                        v->vector[i] = atof(tok);
                }

                /* add word vector to vector hash */
                HASH_ADD_KEYPTR(hh, *vecs, v->word, strlen(v->word), v);
        }

        fclose(fd);

        return;

error_out:
        perror("[populate_vector_hash()]");
        return;
}

/*
 * Construct a vector similarity matrix.
 */

DMat construct_sm(struct config *cfg, struct vectors **vecs)
{
        // DMat dsm = svdNewDMat(cfg->rows, cfg->rows);

        /*
         * 03/03/17: Intializing this matrix with svdNewDMat() from SVDLIBC 
         * blows up if the matrix gets too large, for whatever reason ...
         *
         * Workaround: Initialize the matrix ourselves.
         */

        DMat dsm;
        if(!(dsm = malloc(sizeof(struct dmat))))
                goto error_out;
        memset(dsm, 0, sizeof(struct dmat));

        dsm->rows = cfg->rows;
        dsm->cols = cfg->rows;

        /* allocate rows */
        dsm->value = malloc(dsm->rows * sizeof(double *));
        memset(dsm->value, 0, dsm->rows * sizeof(double *));
        
        /* allocate columns */
        for (int i = 0; i < dsm->rows; i++) {
                if(!(dsm->value[i] = malloc(dsm->cols * sizeof(double))))
                        goto error_out;
                memset(dsm->value[i], 0, dsm->cols * sizeof(double));
        }

        /* end of work around */

        int r; struct vectors *rv;
        for (r = 0, rv = *vecs; r < cfg->rows && rv != NULL; r++, rv = rv->hh.next) {
                double rv_mean = vector_mean(rv->vector, cfg->cols);
                int c; struct vectors *cv;

                for (c = 0, cv = *vecs; c < cfg->rows && cv != NULL; c++, cv = cv->hh.next) {
                        double cv_mean = vector_mean(cv->vector, cfg->cols);
                        dsm->value[r][c] = vector_similarity(rv->vector, rv_mean,
                                        cv->vector, cv_mean, cfg->cols);
                }
        }

        return dsm;

error_out:
        perror("[construct_sm()]");
        return NULL;
}

/*
 * Compute the mean value of a vector:
 *
 * mean = 1/n sum_i v_i
 *
 * where n is the number of dimensions of the vector.
 */

double vector_mean(double *v, int dims)
{
        double mean = 0.0;

        for (int i = 0; i < dims; i++)
                mean += v[i];

        return mean / dims;
}

/*
 * Compute vector similarity using Pearson's correlation.
 *
 *                      sum_i (a_i - a) (b_i - b)
 * S(a,b) = -------------------------------------------------
 *          (sum_i (a_i - a) ^ 2 * sum_i (b_i - b) ^ 2) ^ 0.5
 */

double vector_similarity(double *v1, double v1_mean, double *v2,
                double v2_mean, int dims)
{
        double nom = 0.0, asq = 0.0, bsq = 0.0;

        for (int i = 0; i < dims; i++) {
                nom += (v1[i] - v1_mean) * (v2[i] - v2_mean);
                asq += pow(v1[i] - v1_mean, 2.0);
                bsq += pow(v2[i] - v2_mean, 2.0);
        }

        // return nom / pow(asq * bsq, 0.5);
        double denom = pow(asq * bsq, 0.5);
        if (denom > 0.0)
                return nom / denom;
        else
                return 0.0;
}

/*
 * Write top-k most similar words.
 */

void write_topk_similar_words(struct config *cfg, struct vectors **vecs,
                DMat dsm)
{
        FILE *fd;
        if (!(fd = fopen(cfg->output_fn, "w")))
                goto error_out;

        int r; struct vectors *rv;
        for (r = 0, rv = *vecs; r < cfg->rows && rv != NULL; r++, rv = rv->hh.next) {
                /* identify top-k most similar */
                int topk_idxs[cfg->top_k];
                for (int i = 0; i < cfg->top_k; i++) {
                        topk_idxs[i] = -1;
                        for (int c = 0; c < cfg->rows; c++) {
                                /* skip identical word pairs */
                                if (r == c)
                                        continue;
                                /* skip if we already have this word */
                                bool seen = false;
                                for (int j = 0; j < i; j++)
                                        if (topk_idxs[j] == c)
                                                seen = true;
                                if (seen)
                                        continue;
                                if (topk_idxs[i] == -1)
                                        topk_idxs[i] = c;
                                else if (dsm->value[r][c] > dsm->value[r][topk_idxs[i]])
                                        topk_idxs[i] = c;
                        }
                }

                /* write top-k most similar */
                for (int i = 0; i < cfg->top_k; i++) {
                        int c; struct vectors *cv;
                        for (c = 0, cv = *vecs; c < cfg->rows && cv != NULL; c++, cv = cv->hh.next)
                                if (topk_idxs[i] == c)
                                        fprintf(fd, "%s %s %f\n", rv->word, cv->word, dsm->value[r][c]);
                }
        }

        fclose(fd);

        return;

error_out:
        perror("[write_topk_similar_words()]");
        return;
}
