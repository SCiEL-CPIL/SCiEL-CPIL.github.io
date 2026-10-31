import numpy as np
import time
import zlib
import math
from functools import lru_cache

# Global state to track the current set and precomputed structures
# Since priority() is called repeatedly, we must maintain state across calls
# or compute on the fly if state is not allowed.
# The task says "vectors ... are added greedily".
# Usually, in these FunSearch tasks, the evaluator calls priority(vector) 
# to get a score, then picks the max, adds it, and repeats.
# We cannot assume we can modify a global set inside priority() because 
# priority() should be a pure function of the vector and n for sorting, 
# UNLESS the framework allows stateful functions.
# However, the standard FunSearch contract for cap sets usually implies:
# 1. Generate all vectors (or a subset).
# 2. Sort by priority.
# 3. Iterate: if vector can be added (no conflict with current set), add it.
# 
# If priority() is static, it doesn't know the current set.
# If priority() is dynamic, it needs access to the current set.
# The prompt says: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set".
# This implies the evaluator does:
# S = set()
# for v in sorted(all_vectors, key=priority, reverse=True):
#     if is_cap_set(S U {v}):
#         S.add(v)
#
# In this model, priority(v) is STATIC. It cannot depend on S.
# Therefore, the "global spectral" idea must be pre-computed or static.
# BUT, the inspiration mentions "global spectral representation".
# If priority is static, how can it be global?
# It can use properties of the vector itself that correlate with being "good" 
# in a cap set context, such as being part of a large cap set structure 
# like a hyperplane or a subspace.
#
# A known large cap set in F_3^n is a hyperplane. 
# Vectors in a hyperplane satisfy a linear equation sum(a_i * v_i) = c.
# If we pick a random hyperplane, vectors in it are "safe" with each other 
# (they don't form lines within the hyperplane? No, lines exist in hyperplanes).
# Wait, a hyperplane in F_3^n has size 3^(n-1). 
# Does a hyperplane contain a line? Yes, any affine subspace of dimension >=1 contains lines.
# So a hyperplane is NOT a cap set.
#
# The largest known cap sets are related to the "ellipse" or "oval" constructions 
# or the Croot-Sisask bound constructions.
# For small n, the best cap sets are often constructed via greedy or specific algebraic structures.
#
# Let's look at the "spectral" inspiration again. 
# "scoring vectors based on the magnitude of their Discrete Fourier Transform over Z_3^n"
# The DFT of a vector v in Z_3^n is related to its frequency content.
# Vectors with "low frequency" content (smooth) might be better?
# Or vectors with high frequency?
#
# Actually, a static heuristic that works well for cap sets is to prefer vectors 
# that are "far" from the origin or have a specific weight distribution.
# The current program balances counts.
#
# Alternative Static Heuristic: 
# Prefer vectors that are part of a "cap set friendly" structure.
# One such structure is the set of vectors with sum(v) = 0 mod 3? No, that's a hyperplane, not a cap set.
#
# Let's try a heuristic based on the "line-free" property of the vector itself? 
# A single vector never forms a line.
#
# What if we use a hash-based tie breaker that correlates with algebraic structure?
#
# Let's look at Inspiration 1: "Replace the local coordinate/entropy statistics with a global spectral representation... uses math, numpy, time, zlib"
#
# If I cannot use the current set S in priority(), I must find a static property.
# However, some FunSearch implementations allow the priority function to be stateful 
# if the evaluator passes the current set or if the function is called in a specific way.
# But the signature is `priority(vector: tuple, n: int) -> float`.
#
# If it is strictly static, the best we can do is a "good" ordering.
# The current program scores 1.2227.
# Inspiration 1 scored 1.2930 using "global spectral representation".
# This suggests that the "spectral" feature is a static property of the vector 
# that correlates with being in a large cap set.
#
# The DFT of a vector v in Z_3^n:
# DFT(k) = sum_{v} v * omega^{k.v} ... no, that's for the indicator function.
# For a single vector, the "spectrum" is just its coordinates in the Fourier basis.
# Maybe it means: Compute the DFT of the vector's indicator? No, it's a single vector.
#
# Perhaps it means: The vector v defines a character. 
# Or, it uses the fact that cap sets have low "additive energy".
#
# Let's try a different static feature: 
# The number of "conflicts" a vector would have with a "typical" set?
#
# Actually, there is a known result: 
# The greedy algorithm with a random tie-breaker often performs well.
# The current program has a deterministic tie-breaker based on sum.
#
# Let's try a heuristic that favors vectors with "balanced" sums in a non-linear way.
# Or, use a hash of the vector to break ties, but weight the main score by 
# a feature that is known to be good for cap sets.
#
# One strong feature: 
# Vectors with an even number of 1s and 2s? 
#
# Let's look at the "zlib" import in Inspiration 1. 
# This suggests hashing or compression-based features.
#
# Strategy: 
# 1. Calculate a "spectral" score based on the vector's coordinates treated as a signal.
#    e.g., sum of squares of coordinates? (Always n)
#    sum of products of adjacent coordinates?
#
# Let's try: 
# Score = - (sum of squared differences from mean) + tie_breaker
# This is similar to the current one.
#
# What if we use the "rank" of the vector in a specific ordering?
#
# Let's try a completely different static heuristic: 
# Prefer vectors that are "linearly independent" from a fixed basis?
# No, that's hard to compute statically for arbitrary n.
#
# Let's go with the "Spectral" idea from Inspiration 1.
# Interpret "DFT magnitude" as the energy in the frequency domain.
# For a vector v in {0,1,2}^n, we can map it to {1, -1, -1} or similar?
#
# Actually, a simple and effective heuristic for cap sets is to prefer vectors 
# with a "uniform" distribution of symbols, but with a twist:
# Prefer vectors where the sum of coordinates is 0 mod 3?
# No, that's a hyperplane.
#
# Let's try: 
# Score = - (variance of coordinates) + 0.1 * (hash(vector) / max_hash)
#
# Wait, the current program already does balance.
#
# New Idea: 
# Use the "weight" of the vector in the sense of the Hamming weight of non-zero entries.
# Cap sets often have vectors of various weights.
#
# Let's try a heuristic based on the "linearity" of the vector.
# A vector v is "linear" if it lies on a line with the origin? No.
#
# Let's implement a heuristic that uses a pre-computed "goodness" table for small n?
# No, n is variable.
#
# Let's try the "Spectral" approach:
# Map v to a complex vector using the cube roots of unity: w = exp(2*pi*i/3).
# v_complex = [w^v_i for v_i in v]
# Compute the DFT of v_complex? 
# The DFT of a single vector is just the vector itself in the Fourier basis.
# The magnitude of the "DC" component (sum) is |sum(w^v_i)|.
# If sum(w^v_i) is small, the vector is "balanced" in the complex plane.
# This is a stronger balance measure than just counts.
#
# Let's implement this:
# 1. Convert vector to complex numbers using omega = exp(2*pi*i/3).
# 2. Sum them up.
# 3. The magnitude of this sum is a measure of imbalance.
#    - If all 0: sum = n. Mag = n.
#    - If balanced 0,1,2: sum ~ 0. Mag ~ 0.
#    - If 0,0,1: sum = 1 + 1 + w = 2+w. Mag = |2+w| = sqrt(3).
#
# So, we want to MINIMIZE the magnitude of the sum of w^v_i.
# This is a smooth, global measure of balance.
#
# Additionally, add a small tie-breaker using a hash to ensure determinism and avoid ties.

def priority(vector, n):
    # Precompute omega
    # omega = -0.5 + 0.8660254037844386j
    # For speed, use real/imag parts directly
    # w^0 = 1 + 0j
    # w^1 = -0.5 + 0.8660254037844386j
    # w^2 = -0.5 - 0.8660254037844386j
    
    real_sum = 0.0
    imag_sum = 0.0
    
    # Constants for speed
    cos_1 = -0.5
    sin_1 = 0.8660254037844386
    
    for v in vector:
        if v == 0:
            real_sum += 1.0
        elif v == 1:
            real_sum += cos_1
            imag_sum += sin_1
        else: # v == 2
            real_sum += cos_1
            imag_sum -= sin_1
            
    # Magnitude squared of the sum
    # We want to minimize this.
    mag_sq = real_sum * real_sum + imag_sum * imag_sum
    
    # Base score: negative magnitude squared
    score = -mag_sq
    
    # Tie breaker: use a fast hash of the vector
    # This ensures that vectors with similar balance are ordered deterministically
    # but pseudo-randomly, which can help greedy escape local optima.
    # Using zlib.crc32 is fast and deterministic.
    # Convert vector to bytes for hashing
    # vector is a tuple of ints 0,1,2
    # We can pack it into a string or bytes
    
    # Fast hash: sum of (v_i * 3^i) mod some large prime?
    # Or just use the tuple directly in a hash function.
    # hash(vector) is fast but not stable across runs? 
    # Python's hash() is stable for strings/bytes, but for tuples of ints, 
    # it depends on the order and values. It is deterministic within a run.
    # To be safe and stable, use a simple polynomial hash.
    
    h = 0
    base = 3
    # Simple polynomial hash
    for v in vector:
        h = (h * base + v) % 1000003
    
    # Normalize hash to [0, 1)
    tie_breaker = (h / 1000003.0) * 0.01
    
    return score + tie_breaker
