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
 * Hash table of words for which to ensure inclusion.
 */

struct words
{
        char *word;             /* word string (hash-key) */
        UT_hash_handle hh;      /* hash handle */
};

/*
 * ########################################################################
 * ## Function protoypes                                                 ##
 * ########################################################################
 */

void cprintf(const char *fmt, ...);
void mprintf(const char *fmt, ...);

bool config_is_sane(struct config *cfg);
void print_help(char *exec_name);
void print_version();

void coals(struct config *cfg);

void populate_freq_hash(char *fn, struct freqs **fqs);

void dispose_freq_hash(struct freqs **fqs);

int sort_by_freq(struct freqs *a, struct freqs *b);

/*
 * ########################################################################
 * ## Main functions                                                     ##
 * ########################################################################
 */

int main(int argc, char **argv)
{
        cprintf("");
        cprintf("COALS version %s", VERSION);
        cprintf("Copyright (c) 2013 Harm Brouwer <me@hbrouwer.eu>");
        cprintf("Center for Language and Cognition, University of Groningen");
        cprintf("Netherlands Organisation for Scientific Research (NWO)");
        cprintf("");

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

void cprintf(const char *fmt, ...)
{
        va_list args;

        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

void mprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "--- ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
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
        cprintf(
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

                        "\n"
                        "  basic information for users:\n"
                        "    --help\t\tshows this help message\n"
                        "    --version\t\tshows version\n"
                        "\n",
                        exec_name);
}

void print_version()
{
        cprintf("%s\n", VERSION);
}

/*
 * Main COALS function.
 */

void coals(struct config *cfg)
{
        mprintf("starting construction of COALS vectors:");

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

        /*
         * Populate frequency hash.
         */
        struct freqs *fqs = NULL;
        mprintf("populating frequency hash from: [%s] (...)", cfg->unigrams_fn);
        populate_freq_hash(cfg->unigrams_fn, &fqs);

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
 * where the first integegers denotes that the frequency is
 * is unigram count.
 */

void populate_freq_hash(char *fn, struct freqs **fqs)
{
        FILE *fd;
        if (!(fd = fopen(fn, "r")))
                goto error_out;

        /* read all unigrams */
        char buf[1024];
        while (fgets(buf, sizeof(buf), fd)) {
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
 * Sort hash on basis of frequency (descending).
 */

int sort_by_freq(struct freqs *a, struct freqs *b)
{
        return b->freq - a->freq;
}
