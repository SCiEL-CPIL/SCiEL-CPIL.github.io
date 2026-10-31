"""Constructing a kissing configuration in 11 dimensions using a binary-to-integer mapping."""

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.special import comb


def construct():
    # Vectors of the form (a, b, c, d) where a+b+c+d = 0, all are non-negative integers
    # This is a known construction for kissing numbers in certain dimensions.
    # For D=11, we can map a 4-dimensional lattice to 11 dimensions.
    # But let's try a different approach: use the fact that the kissing number in 11D is known to be at least 518.
    # We can construct vectors by taking all integer vectors with norm squared 2 (the seed approach) which gives 220 vectors.
    # To get more, we can use vectors with norm squared 4.
    # For example, vectors with two 1s and two -1s, or one 2 and one -2, etc.
    # Let's try to construct vectors with norm squared 4.
    # A vector with two 1s and two -1s has norm squared 4.
    # We can choose 4 positions out of 11, and assign signs such that two are +1 and two are -1.
    # The number of such vectors is C(11,4) * 6 (since there are 6 ways to assign two +1s and two -1s to 4 positions).
    # C(11,4) = 330, so 330 * 6 = 1980 vectors.
    # But we need to ensure that the minimum pairwise distance is at least the maximum norm (which is 2).
    # The distance between two such vectors can be as low as 0 (if they are the same) or higher.
    # We need to filter out pairs that are too close.
    # This is a large set, so we might not be able to include all of them.
    # Let's try a simpler construction: use vectors with exactly one non-zero entry, which is ±2.
    # There are 11 positions and 2 signs, so 22 vectors. Norm squared is 4, norm is 2.
    # The distance between any two such vectors is at least 2 (if they are in different positions) or 4 (if they are in the same position but opposite signs).
    # So this gives 22 vectors, which is less than the seed's 220.
    # Let's try vectors with one +1 and one -1 (norm squared 2) and also vectors with one +2 (norm squared 4).
    # But the norm is different, so the condition is min distance >= max norm.
    # Max norm is 2. Min distance between a norm-2 vector and a norm-4 vector can be 1 (e.g., (1,0,...) and (2,0,...) have distance 1).
    # So we can't mix them.
    
    # Let's stick with the seed construction, which is known to be valid.
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
