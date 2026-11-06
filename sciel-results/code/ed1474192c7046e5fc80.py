import numpy as np
import time
from itertools import combinations

def construct():
    # Target N values in descending order
    targets = [34, 33, 32, 31, 30]
    
    start_time = time.time()
    limit = 110  # Safety margin
    
    for n in targets:
        if time.time() - start_time > limit:
            break
            
        # Precompute indices
        triples = list(combinations(range(n), 3))
        E = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Quads (4-sets) -> 4 triples each
        quad_list = list(combinations(range(n), 4))
        Q = np.zeros((len(quad_list), 4), dtype=np.int32)
        for i, q in enumerate(quad_list):
            # q is (a,b,c,d)
            # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            Q[i, 0] = triple_to_idx[(q[0], q[1], q[2])]
            Q[i, 1] = triple_to_idx[(q[0], q[1], q[3])]
            Q[i, 2] = triple_to_idx[(q[0], q[2], q[3])]
            Q[i, 3] = triple_to_idx[(q[1], q[2], q[3])]
            
        # Pentas (5-sets) -> 10 triples each
        penta_list = list(combinations(range(n), 5))
        P = np.zeros((len(penta_list), 10), dtype=np.int32)
        for i, p in enumerate(penta_list):
            # p is (a,b,c,d,e)
            # Triples: abc, abd, abe, acd, ace, ade, bcd, bce, bde, cde
            a,b,c,d,e = p
            P[i, 0] = triple_to_idx[(a,b,c)]
            P[i, 1] = triple_to_idx[(a,b,d)]
            P[i, 2] = triple_to_idx[(a,b,e)]
            P[i, 3] = triple_to_idx[(a,c,d)]
            P[i, 4] = triple_to_idx[(a,c,e)]
            P[i, 5] = triple_to_idx[(a,d,e)]
            P[i, 6] = triple_to_idx[(b,c,d)]
            P[i, 7] = triple_to_idx[(b,c,e)]
            P[i, 8] = triple_to_idx[(b,d,e)]
            P[i, 9] = triple_to_idx[(c,d,e)]
            
        # Initialize colors
        C = np.random.randint(0, 2, size=E, dtype=np.int8)
        
        best_C = C.copy()
        best_score = float('inf')
        
        # Local Search
        t_start = time.time()
        while time.time() - t_start < (limit - (time.time() - start_time) - 5):
            # Check violations
            # Red K4: all 4 triples are 0
            red_mask = (C[Q] == 0).all(axis=1)
            num_red = np.sum(red_mask)
            
            # Blue K5: all 10 triples are 1
            blue_mask = (C[P] == 1).all(axis=1)
            num_blue = np.sum(blue_mask)
            
            score = num_red + num_blue
            if score < best_score:
                best_score = score
                best_C = C.copy()
                
            if score == 0:
                # Found a valid coloring
                return {'n': n, 'colors': best_C.tolist()}
                
            # Pick a flip
            if num_red > 0:
                # Pick a random red K4
                idx = np.random.choice(np.where(red_mask)[0])
                # Pick a random triple from this K4
                t_idx = Q[idx, np.random.randint(0, 4)]
                C[t_idx] = 1 # Flip to Blue
            else:
                # Pick a random blue K5
                idx = np.random.choice(np.where(blue_mask)[0])
                # Pick a random triple from this K5
                t_idx = P[idx, np.random.randint(0, 10)]
                C[t_idx] = 0 # Flip to Red
                
        # If loop finishes without finding 0, try next N
        # We could return best_C if it's better than previous, but we need VALIDITY.
        # So we only return if score == 0.
        
    # Fallback: Return the seed program's result (N=9)
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
