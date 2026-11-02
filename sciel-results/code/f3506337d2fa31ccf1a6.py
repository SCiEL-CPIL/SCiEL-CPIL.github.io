import time as _time
import random
import numpy as np
import math

def construct(k, g):
    _deadline = _time.time() + 110.0 # Leave 10s margin
    
    # 1. Generate non-backtracking words of length 3 to 15
    unique_words = set()
    
    initial_states = []
    for gen in [0, 1, 2]:
        for sign in [1, -1]:
            c1 = sign if gen == 0 else 0
            ca = sign if gen == 1 else 0
            cb = sign if gen == 2 else 0
            initial_states.append((gen, sign, c1, ca, cb, 1))
    
    current_states = initial_states
    
    for length in range(1, 15):
        next_states = []
        for state in current_states:
            last_gen, last_sign, c1, ca, cb, l = state
            
            if l >= 3:
                unique_words.add((c1, ca, cb))
            
            if l == 15:
                continue
                
            for next_gen in [0, 1, 2]:
                for next_sign in [1, -1]:
                    if next_gen == last_gen and next_sign == -last_sign:
                        continue
                    nc1 = c1 + (next_sign if next_gen == 0 else 0)
                    nca = ca + (next_sign if next_gen == 1 else 0)
                    ncb = cb + (next_sign if next_gen == 2 else 0)
                    next_states.append((next_gen, next_sign, nc1, nca, ncb, l + 1))
        current_states = next_states
        if _time.time() > _deadline - 5:
            break

    if not unique_words:
        return None, []

    word_list = list(unique_words)
    W = len(word_list)
    word_arr = np.array(word_list, dtype=np.int16)
    c1_arr = word_arr[:, 0]
    ca_arr = word_arr[:, 1]
    cb_arr = word_arr[:, 2]
    
    max_N = 1000
    
    for N in range(512, max_N + 1, 2):
        if _time.time() > _deadline:
            break
            
        # Find coprime a
        a_candidates = []
        for a in range(2, N//2):
            if math.gcd(a, N) == 1:
                a_candidates.append(a)
        
        for a in a_candidates:
            if _time.time() > _deadline:
                break
                
            b_candidates = []
            for b in range(a + 1, N//2):
                if math.gcd(b, N) == 1:
                    b_candidates.append(b)
            
            if not b_candidates:
                continue
                
            b_vals = np.array(b_candidates, dtype=np.int16)
            M = len(b_vals)
            
            # Chunking to avoid huge memory spikes?
            # Let's try direct first.
            # W ~ 30,000, M ~ 100 (if N is highly composite).
            # If N is prime, M ~ 250.
            # 30,000 * 250 = 7.5M elements. 15MB.
            # This is small!
            
            term1 = c1_arr[:, None]
            term2 = (ca_arr * a)[:, None]
            term3 = cb_arr[:, None] * b_vals[None, :]
            
            vals = term1 + term2 + term3
            mods = vals % N
            
            valid_mask = np.all(mods != 0, axis=0)
            
            if np.any(valid_mask):
                idx = np.where(valid_mask)[0][0]
                b = b_vals[idx]
                
                edges = []
                for i in range(N):
                    for s in [1, a, b]:
                        j = (i + s) % N
                        if i < j:
                            edges.append((i, j))
                
                return N, edges
    
    return None, []
