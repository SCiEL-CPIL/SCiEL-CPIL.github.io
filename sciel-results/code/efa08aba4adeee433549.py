import time
import random
from itertools import combinations

def is_sidon_fast(S):
    """Check if a list of integers is a Sidon set using bitmask."""
    mask = 0
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            bit = 1 << s
            if mask & bit:
                return False
            mask |= bit
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest possible down to smallest
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon_fast(subset):
                return k
    return 2 # Any 2 distinct elements form a Sidon set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    candidates = []
    
    # 1. Pair Sum Structure (Inspiration 3)
    # Pairs summing to K
    for K in range(20, 40):
        # We need 6 pairs.
        # i, K-i. Distinct integers in [0, 128].
        # i < K-i => i < K/2.
        # We need 6 distinct i's such that K-i are distinct and not equal to any i.
        # If K is even, i=K/2 is a singleton. Avoid it or handle carefully.
        # Let's just pick 6 i's from 0 to K//2 - 1.
        if K % 2 == 0:
            max_i = K // 2 - 1
        else:
            max_i = K // 2
            
        if max_i >= 5:
            # Pick i = 0, 1, 2, 3, 4, 5
            # Ensure K-i are distinct and not in {0..5}
            # If K=22, i=0..5, K-i = 22..17. Distinct.
            S = []
            valid = True
            for i in range(6):
                a = i
                b = K - i
                if a == b:
                    valid = False
                    break
                S.append(a)
                S.append(b)
            if valid and len(set(S)) == 12 and max(S) <= 128:
                candidates.append(sorted(S))
                
    # 2. Arithmetic Progressions
    for d in range(1, 11):
        for a in range(0, 128 - 11*d + 1):
            S = [a + i*d for i in range(12)]
            candidates.append(S)
            
    # 3. Random Dense Sets
    for _ in range(10):
        S = random.sample(range(0, 30), 12)
        S.sort()
        candidates.append(S)
        
    best_set = None
    best_score = 13
    
    # Evaluate initial candidates
    for S in candidates:
        if time.time() > deadline: break
        score = max_sidon_size(S)
        if score < best_score:
            best_score = score
            best_set = S[:]
            
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
        
    S = best_set[:]
    
    # Local Search
    while time.time() < deadline:
        improved = False
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try moving S[i] to a random value in a dense range
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 30)
                if new_val in S:
                    continue
                
                new_S = S[:]
                new_S[i] = new_val
                new_S.sort()
                
                score = max_sidon_size(new_S)
                
                if score < best_score:
                    best_score = score
                    S = new_S
                    best_set = S[:]
                    improved = True
                    break
                elif score == best_score and random.random() < 0.05:
                    S = new_S
                    best_set = S[:]
            
            if improved:
                indices = list(range(12))
                random.shuffle(indices)

    return best_set
