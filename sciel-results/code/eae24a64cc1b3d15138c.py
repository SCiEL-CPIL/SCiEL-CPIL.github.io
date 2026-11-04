import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_vectors = []
    best_score = 0
    
    # Seed solution: Weight 2, norm sqrt(2)
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(v)
    best_vectors = seed_vectors
    best_score = len(seed_vectors)
    
    # Candidate generation function
    def generate_candidates(norm_sq, max_weight):
        cands = []
        # Generate all vectors with integer entries, norm_sq, weight <= max_weight
        # This is inefficient if done naively, but D=11 is small.
        # Better: generate supports and signs.
        
        # For norm_sq=5, weight 2: (1, 2) or (2, 1)
        # For norm_sq=6, weight 3: (1, 1, 2)
        
        if norm_sq == 5:
            # Weight 2
            for i, j in itertools.combinations(range(D), 2):
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        v = [0] * D
                        v[i] = s1
                        v[j] = s2 * 2
                        cands.append(tuple(v))
                        v2 = [0] * D
                        v2[i] = s1 * 2
                        v2[j] = s2
                        cands.append(tuple(v2))
        elif norm_sq == 6:
            # Weight 3: 1, 1, 2
            for i, j, k in itertools.combinations(range(D), 3):
                # Permutations of (1, 1, 2)
                # (1, 1, 2), (1, 2, 1), (2, 1, 1)
                for perm in [(1, 1, 2), (1, 2, 1), (2, 1, 1)]:
                    for s1 in (1, -1):
                        for s2 in (1, -1):
                            for s3 in (1, -1):
                                v = [0] * D
                                v[i] = perm[0] * s1
                                v[j] = perm[1] * s2
                                v[k] = perm[2] * s3
                                cands.append(tuple(v))
        return cands

    def greedy_select(candidates, threshold, rng):
        # Random shuffle
        cands = list(candidates)
        rng.shuffle(cands)
        selected = []
        for v in cands:
            if time.time() > deadline:
                break
            ok = True
            for u in selected:
                dot = sum(v[i] * u[i] for i in range(D))
                if dot > threshold:
                    ok = False
                    break
            if ok:
                selected.append(v)
        return selected

    # Try norm sqrt(5)
    cands_5 = generate_candidates(5, 2)
    # Threshold for norm sqrt(5): v.w <= 2.5 -> 2
    for _ in range(50): # 50 restarts
        if time.time() > deadline:
            break
        rng = random.Random(_ + 100)
        sel = greedy_select(cands_5, 2, rng)
        if len(sel) > best_score:
            best_score = len(sel)
            best_vectors = sel
            
    # Try norm sqrt(6)
    cands_6 = generate_candidates(6, 3)
    # Threshold for norm sqrt(6): v.w <= 3
    for _ in range(20): # 20 restarts, larger pool
        if time.time() > deadline:
            break
        rng = random.Random(_ + 200)
        sel = greedy_select(cands_6, 3, rng)
        if len(sel) > best_score:
            best_score = len(sel)
            best_vectors = sel
            
    return np.array(best_vectors, dtype=np.int64)
