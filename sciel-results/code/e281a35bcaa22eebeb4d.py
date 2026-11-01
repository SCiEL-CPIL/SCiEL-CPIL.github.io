import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Type A: +/- 2 * e_i
    # These have norm squared 4.
    type_a = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            type_a.append(v)
            
    # Type B: +/- e_i +/- e_j +/- e_k +/- e_l
    # These have norm squared 4.
    type_b = []
    indices = range(D)
    for support in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            type_b.append(v)
            
    # Total candidates: 22 + 5280 = 5302
    # We want to select a subset S such that min pairwise dist sq >= 4 (since max norm sq is 4).
    # Actually, max norm is 2. Min dist must be >= 2. Dist sq >= 4.
    
    # Check compatibility
    def is_compatible(v, S):
        for u in S:
            # dist sq = ||v-u||^2 = ||v||^2 + ||u||^2 - 2 v.u = 4 + 4 - 2 v.u = 8 - 2 v.u
            # We need dist sq >= 4 => 8 - 2 v.u >= 4 => 2 v.u <= 4 => v.u <= 2
            dot = np.dot(v, u)
            if dot > 2:
                return False
        return True
        
    best_set = []
    best_size = 0
    
    # Start with Type A, they are mutually compatible?
    # u=2e_i, v=2e_j. dot=0. OK.
    # u=2e_i, v=-2e_i. dot=-4. OK.
    # So all Type A are compatible with each other.
    current_set = list(type_a)
    
    # Greedy add Type B
    # We can try multiple random orderings
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        random.shuffle(type_b)
        S = list(current_set)
        
        for v in type_b:
            if time.time() > deadline:
                break
            if is_compatible(v, S):
                S.append(v)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Convert to array
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
