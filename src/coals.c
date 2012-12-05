/*
 * coals.c
 *
 * Copyright 2012 Harm Brouwer <me@hbrouwer.eu>
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

#include "../lib/uthash-1.9.7/src/uthash.h"
#include "../lib/SVDLIBC/svdlib.h"

#define VERSION "0.1-beta"

#define WTYPE_RAMPED 0
#define WTYPE_FLAT 1

#define VTYPE_REAL 0
#define VTYPE_BINARY 1

/*
 * This implements the Correlated Occurrence Analogue to Lexical Semantics
 * (COALS; Rohde, Gonnerman, and Plaut, 2005).
 *
 * (description goes here)
 *
 * References
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

};

/*
 * Word and co-occurrence frequency hash table.
 */

struct freqs
{
        char *word;             /* word string (hash-key) */
        int freq;               /* word frequency */
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

/*
 * ########################################################################
 * ## Main functions                                                     ##
 * ########################################################################
 */

int main(int argc, char **argv)
{
        printf("\n");
        printf("COALS version %s\n", VERSION);
        printf("Copyright (c) 2013 Harm Brouwer <me@hbrouwer.eu>\n");
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
                                        cfg->v_type = WTYPE_RAMPED;
                                if (strcmp(argv[i], "flat") == 0)
                                        cfg->v_type = WTYPE_FLAT;
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

        coals(cfg);

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
        if (cfg->output_fn == NULL)
                return false;

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
                        "    --wtype <type>\tuse <type> window for co-occurrences (default: ramped)\n"
                        "    --rows <num>\tset number of rows of the co-occurrence matrix to <num>\n"
                        "    --cols <num>\tset number of cols of the co-occurrence matrix to <num>\n"
                        "    --dims <num>\treduce COALS vectors to <num> dimensions (using SVD)\n"
                        "    --vtype <type>\tconstruct <type> (real/binary) vectors (default: real)\n"
                        "    --unigrams <file>\tread unigram counts (word frequencies) from <file>\n"
                        "    --ngrams <file>\tread n-gram counts (co-occurrence freqs.) from <file>\n"
                        "    --output <file>\twrite COALS vectors to <file>\n"
                        "    --enfore <file>\tenforce inclusion of words in <file>\n"

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
        fprintf(stderr, "--- starting construction of COALS vectors\n");

        fprintf(stderr, "\tco-occurrence window size:\t[%d]\n", cfg->w_size);
        fprintf(stderr, "\tco-occurrence window type:\t");
        if (cfg->v_type == WTYPE_RAMPED)
                printf("[ramped]\n");
        if (cfg->v_type == WTYPE_FLAT)
                printf("[flat]\n");

        fprintf(stderr, "\tco-occurrence matrix rows:\t[%d]\n", cfg->rows);
        fprintf(stderr, "\tco-occurrence matrix cols:\t[%d]\n", cfg->cols);

        fprintf(stderr, "\tCOALS vector dimensions:\t[%d]\n", cfg->dims);
        fprintf(stderr, "\tCOALS vector type:\t\t");
        if (cfg->v_type == VTYPE_REAL)
                printf("[real]\n");
        if (cfg->v_type == VTYPE_BINARY)
                printf("[binary]\n");

        fprintf(stderr, "\tunigrams file:\t\t\t[%s]\n", cfg->unigrams_fn);
        fprintf(stderr, "\tngrams file:\t\t\t[%s]\n", cfg->ngrams_fn);
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
        fprintf(stderr, "\tbuilt a [%ldx%ld] matrix\n", dcm->rows, dcm->cols);

        /*
         * Convert frequencies to correlations.
         */
        fprintf(stderr, "--- converting frequencies to correlation coefficients (...)\n");
        convert_freqs_to_correlations(dcm);

        /*
         * Reduce dimensionality (with SVD; if required).
         */
        SMat scm = NULL; SVDRec svdrec = NULL;
        if (cfg->dims > 0) {
                fprintf(stderr, "--- reducing dimensionality with singular value decomposition (...)\n");
                scm = svdConvertDtoS(dcm);
                svdrec = svdLAS2A(scm, cfg->dims);
        }

        /* clean up */
        fprintf(stderr, "--- cleaning up (...)\n");
        dispose_enf_word_hash(&ewds);
        dispose_freq_hash(&fqs);
}

/*
 * Populate word frequencies on basis of unigram counts.
 *
 * This function assumes the following file format:
 *
 * 1|<word_1>|<frequency>
 * 1|<word_2>|<frequency>
 * .|........|...........
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
        char buf[1024];
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
                int freq = atoi(++fp);

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
 * n|<word_1>|<frequency>
 * n|<word_2>|<frequency>
 * .|........|...........
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
        char buf[1024];
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
                int freq = atoi(++fp);

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
                        int rfreq = freq;

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
         * String is a non-word is it contains
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
        return b->freq - a->freq;
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
        char buf[1024];
        while (fgets(buf, sizeof(buf), fd)) {
                /* copy word */
                char *word;
                int block_size = strlen(buf) * sizeof(char);
                if (!(word = malloc(block_size)))
                        goto error_out;
                memset(word, 0, block_size);
                strncpy(word, buf, strlen(buf) - 1); 
                
                /*
                 * We only want to add a word if it occurs in the corpus.
                 * This is the case if we have its frequency.
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
                        fprintf(stderr, "\tno frequency for word:\t\t(%s)\n", word);
                        free(word);
                }
        }

        /* 
         * Delete words that occur in the top-k frequencies;
         * We do not want the same word to occur twice.
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
 * correlations are set to zero, and positive correlations are magnified
 * by taking their square root.
 */

void convert_freqs_to_correlations(DMat dcm)
{
        DMat rts = svdNewDMat(dcm->rows, 1);
        DMat cts = svdNewDMat(1, dcm->cols);
        double t;

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
         * take its square root to magnify it.
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
