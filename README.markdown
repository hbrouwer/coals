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

References
----------

* Rohde, D. L. T., Gonnerman, L. M., & Plaut, D. C. (2005). *An Improved
  Model of Semantic Similarity Based on Lexical Co-Occurrence*.
