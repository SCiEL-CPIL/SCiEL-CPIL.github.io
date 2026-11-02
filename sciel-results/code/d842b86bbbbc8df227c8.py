"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import math
import sys

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or max(U) < 0:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    # Compute |U+U| and |U-U|
    # For efficiency, use sets
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _random_set(n, max_val, rng):
    """Generate a random set of n elements including 0."""
    if n <= 1:
        return [0]
    elements = {0}
    while len(elements) < n:
        elements.add(rng.randint(1, max_val))
    return sorted(elements)

def _local_search(U, max_val, rng, deadline, iters_per_step=50):
    """Local search: try adding/removing elements."""
    U = list(U)
    best_score = _score_set(U)
    
    for _ in range(iters_per_step):
        if time.time() > deadline:
            break
        # Try adding a random element
        if len(U) < 200:
            new_elem = rng.randint(1, max_val)
            if new_elem not in U:
                new_U = U + [new_elem]
                s = _score_set(new_U)
                if s > best_score:
                    U = new_U
                    best_score = s
        # Try removing a random non-zero element
        if len(U) > 2:
            idx = rng.randint(1, len(U) - 1)
            new_U = U[:idx] + U[idx+1:]
            s = _score_set(new_U)
            if s > best_score:
                U = new_U
                best_score = s
        # Try replacing an element
        if len(U) > 2:
            idx = rng.randint(1, len(U) - 1)
            new_elem = rng.randint(1, max_val)
            if new_elem not in U:
                new_U = U[:idx] + [new_elem] + U[idx+1:]
                s = _score_set(new_U)
                if s > best_score:
                    U = new_U
                    best_score = s
    
    return U, best_score

def _structured_set(k):
    """Generate a structured set based on known constructions.
    
    The paper mentions sets of size 2003 and 54265.
    A common construction for sumset/difference-set problems uses
    geometric progressions or mixed radix systems.
    
    Let's try sets based on powers of a base, or combinations.
    """
    # Try a set based on powers of 2 with some structure
    # U = {sum of a_i * b_i where a_i in {0,1}}
    # This gives |U| = 2^k, |U+U| ~ 2^(k+1), |U-U| ~ 2^(k+1)
    # But we want |U-U| > |U+U|
    
    # Try U = {0, 1, M, 1+M, 2M, 1+2M, ...}
    # Or use a base-B expansion where differences are more numerous
    
    # Let's try: U = {i * B + j : i in range(k), j in range(2)}
    # This gives |U| = 2k
    # U+U has elements (i1+i2)*B + (j1+j2) where j1+j2 in {0,1,2}
    # U-U has elements (i1-i2)*B + (j1-j2) where j1-j2 in {-1,0,1}
    
    # For larger B, the B-coefficient dominates, so:
    # |U+U| ~ k * 3 (for the B part, roughly k distinct sums times 3 for the remainder)
    # |U-U| ~ 2k * 3 (for the B part, roughly 2k distinct differences times 3)
    # Ratio ~ 2
    
    # Actually let's be more careful.
    # U = {i*B + j : i=0..k-1, j=0,1}
    # U+U = {(i1+i2)*B + (j1+j2) : i1,i2=0..k-1, j1,j2 in {0,1}}
    # i1+i2 ranges from 0 to 2k-2, so k*2-1 = 2k-1 distinct values
    # j1+j2 in {0,1,2}
    # So |U+U| = (2k-1) * 3 if B > 2
    # U-U = {(i1-i2)*B + (j1-j2) : i1,i2=0..k-1, j1,j2 in {0,1}}
    # i1-i2 ranges from -(k-1) to k-1, so 2k-1 distinct values
    # j1-j2 in {-1,0,1}
    # So |U-U| = (2k-1) * 3
    # Ratio = 1, not helpful.
    
    # Try U = {i*B + j : i=0..k-1, j=0..m-1}
    # |U| = k*m
    # U+U: (i1+i2)*B + (j1+j2), i1+i2 in 0..2k-2 (2k-1 values), j1+j2 in 0..2m-2 (2m-1 values)
    # |U+U| = (2k-1)*(2m-1) if B >= 2m-1
    # U-U: (i1-i2)*B + (j1-j2), i1-i2 in -(k-1)..k-1 (2k-1 values), j1-j2 in -(m-1)..m-1 (2m-1 values)
    # |U-U| = (2k-1)*(2m-1)
    # Ratio = 1 again.
    
    # We need asymmetry. Let's use different ranges for sum and diff.
    # Try U = {i*B + j : i=0..k-1, j=0..m-1} but with B chosen so that
    # carries in addition reduce |U+U| while differences don't have this issue.
    
    # Actually, the key insight is that for difference sets, we get more distinct values
    # because i1-i2 can be negative, while i1+i2 is always non-negative.
    # But the count of distinct i1-i2 is 2k-1 and distinct i1+i2 is 2k-1 too.
    
    # Let me think differently. The score involves log(2*max(U)+1) in the denominator.
    # So we want large |U-U|/|U+U| and small max(U).
    
    # Try a set where elements are sparse: U = {a_i} where a_i grow geometrically.
    # U = {0, 1, B, B^2, ..., B^(k-1)}
    # |U| = k
    # U+U: sums of two powers, at most k(k+1)/2 distinct (if B is large enough)
    # U-U: differences, at most k(k-1) + k = k^2 distinct
    # Ratio ~ k^2 / (k^2/2) = 2
    # max(U) = B^(k-1), so log(2*max+1) ~ (k-1)*log(B)
    # Score ~ 1 + log(2)/((k-1)*log(B))
    # To maximize, minimize k and B.
    
    # With k=3, B=2: U={0,1,2,4}, max=4, log(9)=2.197
    # U+U = {0,1,2,3,4,5,6,8} -> 8 elements
    # U-U = {-4,-3,-2,-1,0,1,2,3,4} -> 9 elements
    # Ratio = 9/8 = 1.125
    # Score = 1 + log(1.125)/log(9) = 1 + 0.1178/2.197 = 1.0536
    
    # With k=4, B=2: U={0,1,2,4,8}, max=8, log(17)=2.833
    # U+U: 0,1,2,3,4,5,6,7,8,9,10,12,16 -> let's count
    # Actually this is getting complex. Let's just compute.
    
    return None

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    rng = random.Random(42)
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try some structured sets
    # Geometric progression
    for B in range(2, 10):
        for k in range(3, 15):
            if time.time() > deadline:
                break
            U = [B**i for i in range(k)]
            if 0 not in U:
                U.append(0)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
        if time.time() > deadline:
            break
    
    # Try mixed structures
    for B in range(2, 20):
        for k in range(2, 10):
            if time.time() > deadline:
                break
            # U = {i*B + j : i=0..k-1, j=0,1}
            U = []
            for i in range(k):
                U.append(i * B)
                U.append(i * B + 1)
            U = sorted(set(U))
            if 0 not in U:
                U.append(0)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
        if time.time() > deadline:
            break
    
    # Random search with local improvement
    max_val = 50
    for n in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]:
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            U = _random_set(n, max_val, rng)
            U, s = _local_search(U, max_val, rng, deadline, iters_per_step=30)
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    # Try larger max_val with smaller sets
    for max_val in [100, 200, 500, 1000]:
        if time.time() > deadline:
            break
        for n in [3, 4, 5, 6, 8, 10]:
            if time.time() > deadline:
                break
            for attempt in range(20):
                if time.time() > deadline:
                    break
                U = _random_set(n, max_val, rng)
                U, s = _local_search(U, max_val, rng, deadline, iters_per_step=20)
                if s > best_score:
                    best_score = s
                    best_U = list(U)
    
    # Final polish: try more local search on the best found
    if time.time() < deadline:
        max_val = max(best_U) + 100
        best_U, best_score = _local_search(best_U, max_val, rng, deadline, iters_per_step=200)
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)
# EVOLVE-BLOCK-END
