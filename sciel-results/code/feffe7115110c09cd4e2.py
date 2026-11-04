import numpy as np
import time
import random
from collections import deque

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution: Random + Local Search (Steepest Descent)
    # This ensures we start at a local optimum, which is better for Tabu Search.
    p = np.random.permutation(n).astype(np.int64)
    
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
        
    def delta_cost(i, j, p):
        li = p[i]
        lj = p[j]
        
        # Vectorized sum S
        # S = a[i, :] . (b[lj, p] - b[li, p]) 
        #   + a[:, i] . (b[p, lj] - b[p, li])
        #   + a[j, :] . (b[li, p] - b[lj, p])
        #   + a[:, j] . (b[p, li] - b[p, lj])
        
        diff_row = b[lj, p] - b[li, p]
        diff_col = b[p, lj] - b[p, li]
        
        S = np.dot(a[i, :], diff_row) + np.dot(a[:, i], diff_col) + \
            np.dot(a[j, :], -diff_row) + np.dot(a[:, j], -diff_col)
            
        # Correction term X + Y
        # X = a[i,j]b[li, lj] + a[j,i]b[lj, li]
        # Y = a[i,j]b[lj, li] + a[j,i]b[li, lj]
        correction = a[i, j] * b[li, lj] + a[j, i] * b[lj, li] + \
                     a[i, j] * b[lj, li] + a[j, i] * b[li, lj]
                     
        return S + correction

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Quick Local Search to polish initial solution
    start_time = time.time()
    time_limit = 115.0
    
    # Steepest Descent Local Search
    improved = True
    while improved and (time.time() - start_time) < 5.0: # Max 5s for init
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                dc = delta_cost(i, j, p)
                if dc < best_delta:
                    best_delta = dc
                    best_i, best_j = i, j
        if best_i != -1:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True

    # 2. Tabu Search
    tabu_tenure = 50
    tabu_list = deque(maxlen=tabu_tenure)
    K = 50  # Candidates per iteration
    
    while time.time() - start_time < time_limit:
        candidates = []
        for _ in range(K):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                candidates.append((i, j))
                
        best_dc = float('inf')
        best_move = None
        best_is_tabu = False
        best_dc_tabu = float('inf')
        
        for i, j in candidates:
            dc = delta_cost(i, j, p)
            key = (min(i, j), max(i, j))
            is_tabu = key in tabu_list
            
            if is_tabu:
                if dc < best_dc_tabu:
                    best_dc_tabu = dc
                    best_dc = dc # Track for aspiration
                    best_move = (i, j)
                    best_is_tabu = True
            else:
                if dc < best_dc:
                    best_dc = dc
                    best_move = (i, j)
                    best_is_tabu = False
                    
        if best_move is None:
            continue
            
        i, j = best_move
        
        # Aspiration Criterion: Accept tabu move if it improves global best
        if best_is_tabu:
            if current_cost + best_dc < best_cost:
                pass # Accept
            else:
                # Reject tabu move, try to find a non-tabu move?
                # For simplicity, if the best candidate is tabu and not aspiration,
                # we might just skip this iteration or pick a random non-tabu one.
                # To keep code simple and fast, we'll just skip if best is tabu and not aspiration.
                # But this might stall. Let's force a random non-tabu move if best is tabu.
                # Re-evaluate: The loop above tracked best_dc (overall) and best_dc_tabu.
                # If best_is_tabu is True, it means the absolute best was tabu.
                # We need the best NON-tabu.
                # Let's restructure the candidate selection slightly.
                
                # Optimization: Keep track of best non-tabu separately.
                # I will modify the logic to ensure we always have a valid move.
                
                # Let's just pick a random valid non-tabu move if the best is tabu and not aspiration.
                found = False
                for _ in range(10): # Try a few randoms
                    ri = random.randint(0, n - 1)
                    rj = random.randint(0, n - 1)
                    if ri == rj: continue
                    rkey = (min(ri, rj), max(ri, rj))
                    if rkey not in tabu_list:
                        i, j = ri, rj
                        best_dc = delta_cost(i, j, p)
                        found = True
                        break
                if not found:
                    # All randoms were tabu? Unlikely with tenure 50 and n=100.
                    # Just pick the tabu one and accept it to break the cycle?
                    # Or skip.
                    continue
        else:
            # Best move was non-tabu, so best_dc is the delta of that move
            # We already have i, j from best_move
            pass

        # Apply move
        p[i], p[j] = p[j], p[i]
        current_cost += best_dc
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update Tabu List
        tabu_list.append((min(i, j), max(i, j)))
        
    return best_p.tolist()
