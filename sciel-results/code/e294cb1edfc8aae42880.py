import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s buffer
    
    # Try n from 34 down to 20
    for n in range(34, 19, -1):
        if time.time() > deadline - 10:  # Need at least 10s for SA
            break
            
        triples = list(combinations(range(n), 3))
        M = len(triples)
        idx_map = {t: i for i, t in enumerate(triples)}
        
        # Precompute indices
        k4_indices = [None] * M
        k5_indices = [None] * M
        
        for i, t in enumerate(triples):
            a, b, c = t
            k4_list = []
            for d in range(n):
                if d == a or d == b or d == c: continue
                verts = sorted([a,b,c,d])
                v1, v2, v3, v4 = verts
                all_triples = [(v1,v2,v3), (v1,v2,v4), (v1,v3,v4), (v2,v3,v4)]
                others = [idx_map[tt] for tt in all_triples if tt != t]
                k4_list.append(others)
            k4_indices[i] = np.array(k4_list)
            
            k5_list = []
            remaining = [v for v in range(n) if v != a and v != b and v != c]
            for e_idx in range(len(remaining)):
                for f_idx in range(e_idx + 1, len(remaining)):
                    e = remaining[e_idx]
                    f = remaining[f_idx]
                    verts = sorted([a,b,c,e,f])
                    all_triples = list(combinations(verts, 3))
                    others = [idx_map[tt] for tt in all_triples if tt != t]
                    k5_list.append(others)
            k5_indices[i] = np.array(k5_list)
            
        # SA
        colors = np.random.randint(0, 2, M)
        
        # Initial Cost
        red_pot = 0
        blue_pot = 0
        for i in range(M):
            if colors[i] == 0:
                red_pot += np.sum(np.all(colors[k4_indices[i]] == 0, axis=1))
            else:
                blue_pot += np.sum(np.all(colors[k5_indices[i]] == 1, axis=1))
        current_cost = red_pot / 4.0 + blue_pot / 10.0
        
        if current_cost == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        T = 1.0
        cooling = 0.995
        while time.time() < deadline - 5:
            t = np.random.randint(M)
            
            if colors[t] == 0:
                r_p = np.sum(np.all(colors[k4_indices[t]] == 0, axis=1))
                b_p = np.sum(np.all(colors[k5_indices[t]] == 1, axis=1))
                delta = (b_p / 10.0) - (r_p / 4.0)
            else:
                r_p = np.sum(np.all(colors[k4_indices[t]] == 0, axis=1))
                b_p = np.sum(np.all(colors[k5_indices[t]] == 1, axis=1))
                delta = (r_p / 4.0) - (b_p / 10.0)
                
            if delta < 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
                colors[t] = 1 - colors[t]
                current_cost += delta
                if current_cost == 0:
                    return {'n': n, 'colors': colors.tolist()}
            T *= cooling
            if T < 0.01: T = 0.01

    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
