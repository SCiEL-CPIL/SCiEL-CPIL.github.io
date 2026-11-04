import numpy as np
import time
from collections import deque

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank Matching
    # Heavy flows (high row sum in A) to short distances (low row sum in B)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    def calc_cost(p):
        B_perm = b[np.ix_(p, p)]
        return np.sum(a * B_perm)

    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu Search Parameters
    tabu_length = 128
    tabu_list = deque()
    start_time = time.time()
    time_limit = 115.0
    
    rng = np.random.default_rng(42)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Select random i, j
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Check Tabu
        # A move is tabu if it reverses a recent move.
        # We store the pair of facility indices (min, max).
        pair = (i, j) if i < j else (j, i)
        
        is_tabu = pair in tabu_list
        
        # Calculate Delta
        diff_a = a[i] - a[j]
        diff_b = b[v, p] - b[u, p]
        S = np.dot(diff_a, diff_b)
        term_i = diff_a[i] * diff_b[i]
        term_j = diff_a[j] * diff_b[j]
        delta = S - term_i - term_j
        
        new_cost = current_cost + delta
        
        # Aspiration Criterion: Accept if it improves best_cost, even if tabu
        if is_tabu and new_cost >= best_cost:
            continue
            
        # Accept the move
        p[i], p[j] = p[j], p[i]
        current_cost = new_cost
        
        # Update Tabu List
        tabu_list.append(pair)
        if len(tabu_list) > tabu_length:
            tabu_list.popleft()
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # Final Polish: Best Improvement Local Search
    # Polish the best solution found
    p = best_p.copy()
    current_cost = best_cost
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                diff_a = a[i] - a[j]
                diff_b = b[v, p] - b[u, p]
                S = np.dot(diff_a, diff_b)
                term_i = diff_a[i] * diff_b[i]
                term_j = diff_a[j] * diff_b[j]
                delta = S - term_i - term_j
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        
        if time.time() - start_time > time_limit:
            break

    return best_p.tolist()
