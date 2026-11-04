import numpy as np
import time
import random

def score_set_numpy(U):
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U: return -1
    n = len(U)
    if n < 2: return -1
    
    # Sumset
    # U + U
    sums = U[:, None] + U[None, :]
    unique_sums = np.unique(sums)
    sum_count = len(unique_sums)
    
    # Diffset
    # U - U
    diffs = U[:, None] - U[None, :]
    unique_diffs = np.unique(diffs)
    diff_count = len(unique_diffs)
    
    if sum_count == 0: return -1
    ratio = diff_count / sum_count
    if ratio <= 1: return -1
    
    max_u = np.max(U)
    D = 2 * max_u + 1
    if D <= 1: return -1
    
    return 1 + np.log(ratio) / np.log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_score = -1
    best_set = [0, 1]
    
    # Strategy: Greedy construction with local search
    # We want to maximize |U-U|/|U+U|.
    # Note that |U-U| is always at least |U+U| if 0 in U? No.
    # |U-U| contains U+U if we flip signs? No.
    # However, if U is symmetric around 0 (i.e., U = V union -V), then U+U and U-U have same size.
    # But U must be non-negative.
    # Let's try to construct a set that is "additively thin" but "subtractively rich".
    # Actually, for any set of non-negative integers, |U-U| >= |U+U| is not necessarily true.
    # Example: U={0,1}. U+U={0,1,2} (size 3). U-U={-1,0,1} (size 3). Ratio 1.
    # U={0,1,3}. U+U={0,1,2,3,4,6} (size 6). U-U={-3,-2,-1,0,1,2,3} (size 7). Ratio 7/6.
    
    # The paper mentions constructions of size 2003 and 54265.
    # We need to find a set that maximizes the exponent.
    # The exponent is roughly log(ratio) / log(maxU).
    # To maximize this, we want ratio to be large and maxU to be small.
    # However, ratio is bounded by something related to the structure.
    
    # Let's try a simple heuristic:
    # Start with {0, 1}.
    # Iteratively add elements that maximize the immediate gain in score.
    # Since full score calculation is expensive, we might approximate or just do it if N is small.
    # But N can be large.
    
    # Let's try a specific known good structure:
    # Powers of 2?
    # U = {0, 1, 2, 4, ...}.
    # Sumset of powers of 2 is all numbers up to 2^(k+1)-1. Size ~ 2^(k+1).
    # Diffset is all numbers from -(2^k-1) to 2^k-1. Size ~ 2^(k+1).
    # Ratio ~ 1. Score ~ 1.
    
    # What about a set with large gaps?
    # U = {0, 1, M, M+1, 2M, 2M+1, ...}
    # This creates a "block" structure.
    
    # Let's try random search with a specific bias towards sparse sets.
    # Or try to optimize a small set first to understand the landscape.
    
    # Given the time constraint, let's try a few candidate structures.
    
    # Candidate 1: Small random sets
    for _ in range(50):
        if time.time() > deadline:
            break
        n = random.randint(10, 50)
        max_val = random.randint(100, 1000)
        U = set([0])
        for _ in range(n-1):
            U.add(random.randint(1, max_val))
        U = sorted(list(U))
        s = score_set_numpy(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Candidate 2: Structured sets (e.g., arithmetic progressions with gaps)
    # U = {0, 1, G, G+1, 2G, 2G+1, ...}
    for G in [10, 20, 50, 100]:
        if time.time() > deadline:
            break
        for k in [2, 3, 4, 5]:
            U = []
            for i in range(k):
                U.append(i*G)
                U.append(i*G + 1)
            U = sorted(list(set(U)))
            s = score_set_numpy(U)
            if s > best_score:
                best_score = s
                best_set = U

    # Candidate 3: "Babai-Frankl" type sets?
    # The paper likely refers to a specific construction.
    # Without the paper, we guess.
    # Often, sets with elements growing exponentially but with some structure yield better ratios.
    
    # Let's try a greedy adder with a cheap proxy score.
    # Proxy: Minimize collisions in sums, maximize distinctness in diffs.
    # This is hard to approximate cheaply.
    
    # Let's just run a bit more random search with different ranges.
    for _ in range(100):
        if time.time() > deadline:
            break
        n = random.randint(20, 100)
        max_val = random.randint(1000, 10000)
        U = set([0])
        # Bias towards smaller numbers to keep maxU small?
        # But we need enough spread for diffset.
        for _ in range(n-1):
            U.add(random.randint(1, max_val))
        U = sorted(list(U))
        s = score_set_numpy(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set
