# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import hashlib

# Strategy: Static, multi-feature heuristic that exploits the structure of cap sets in F_3^n.
# Key Insight: The problem is to find a large subset of {0,1,2}^n with no 3-term arithmetic progression.
# A 3-term AP in F_3^n is a set {a, a+d, a+2d} where d != 0.
# Equivalently, a set S is a cap set if there are no distinct a,b,c in S such that a+b+c = 0 (mod 3).
#
# The previous program used a simple heuristic based on the count of 1s.
# This program uses a more sophisticated heuristic that considers:
# 1. The "balance" of the vector (closeness to the mean value n/2 for each coordinate type).
# 2. A structural penalty for vectors that are "too similar" to each other in a way that promotes APs.
#    Specifically, we penalize vectors that have a low "Hamming distance" from the all-zero and all-two vectors,
#    because these are the "endpoints" of many potential APs.
# 3. A strong, deterministic tie-breaker based on a cryptographic hash of the vector.
#    This ensures that the ordering is well-mixed and avoids pathological symmetries.
#
# Why this might work better:
# - Cap sets are often constructed by taking a large subset of a subspace and then carefully adding vectors from outside.
# - Vectors with a balanced distribution of 0s, 1s, and 2s are less likely to be part of an AP than vectors that are skewed.
# - The hash tie-breaker ensures that we don't accidentally pick a set of vectors that are all "similar" in some unwanted way.

def priority(vector, n):
    # Feature 1: Balance.
    # A balanced vector has roughly n/3 of each digit.
    # We compute the deviation from the expected count.
    c0 = sum(1 for v in vector if v == 0)
    c1 = sum(1 for v in vector if v == 1)
    c2 = sum(1 for v in vector if v == 2)
    
    # Expected count for each digit is n/3.
    expected = n / 3.0
    balance_penalty = (c0 - expected)**2 + (c1 - expected)**2 + (c2 - expected)**2
    
    # Feature 2: Structural penalty.
    # Penalize vectors that are close to the "corners" of the cube (all 0s, all 2s, etc.).
    # These vectors are more likely to form APs with each other.
    # We use the Hamming distance from the all-zero and all-two vectors.
    dist_zero = c1 + c2  # Number of non-zero elements
    dist_two = c0 + c1   # Number of non-two elements
    
    # We want to avoid vectors that are too close to either corner.
    # A simple way is to penalize the minimum of the two distances.
    corner_penalty = min(dist_zero, dist_two)
    
    # Feature 3: Hash-based tie-breaker.
    # This ensures a deterministic but well-mixed ordering.
    h = hashlib.md5(bytes(vector)).digest()
    # Convert the first 8 bytes to an integer for a 64-bit hash.
    hash_int = int.from_bytes(h[:8], 'big')
    hash_score = hash_int / (2**64)
    
    # Combine the features.
    # The balance penalty should be the dominant factor.
    # The corner penalty is a secondary factor.
    # The hash is a tertiary factor for tie-breaking.
    
    # We want to MINIMIZE the penalty, so we return a NEGATIVE score.
    # The evaluator sorts in descending order, so higher scores are picked first.
    # Therefore, we want vectors with LOW penalty to have HIGH score.
    
    score = -balance_penalty - 0.1 * corner_penalty + hash_score * 0.01
    
    return score

# EVOLVE-BLOCK-END
