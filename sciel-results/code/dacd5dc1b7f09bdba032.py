"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # --- Greedy Construction (GRASP-style)
    # We will run a few random greedy constructions to pick the best starting point.
    
    def greedy_construct(rng_seed):
        rng = np.random.default_rng(rng_seed)
        p = np.full(n, -1, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        
        # Precompute row sums for a
        a_row_sum = a.sum(axis=1)
        
        # Sort facilities by total flow to guide the heuristic? 
        # Or just pick randomly? 
        # Let's pick the facility with highest remaining flow impact first, 
        # but allow randomization among top candidates.
        
        # Simple approach: 
        # 1. Calculate incremental cost for each (facility, location) pair given current placement.
        # 2. Pick the best one, or pick randomly from top K.
        
        # To make it faster, we can just use a standard greedy:
        # Place facility i at location l that minimizes sum_{k placed} a[i,k]*b[l,p[k]] + a[k,i]*b[p[k],l]
        
        # Let's use a slightly randomized greedy:
        # At each step, calculate cost for all unplaced facilities and all unused locations.
        # Pick the pair (f, l) with minimum cost. 
        # To add randomness, we can pick from the top 10% of best pairs.
        
        # For efficiency, let's stick to a deterministic greedy first, then perturb it.
        # Actually, the seed's greedy is deterministic. Let's use a multi-start local search.
        
        # Construction:
        # Sort facilities by degree (sum of flows).
        order = np.argsort(-(a.sum(axis=1) + a.sum(axis=0)), kind="stable")
        
        for idx in range(n):
            f = order[idx]
            costs = np.zeros(n, dtype=np.int64)
            for loc in range(n):
                if used[loc]:
                    costs[loc] = np.iinfo(np.int64).max
                    continue
                c = 0
                for k in range(idx):
                    g = order[k]
                    loc_g = p[g]
                    c += a[f, g] * b[loc, loc_g] + a[g, f] * b[loc_g, loc]
                costs[loc] = c
            
            # Randomized choice: pick from top 5 lowest costs
            # But for n=100, this is cheap enough.
            # Let's just pick the best one to ensure a good start.
            best_loc = np.argmin(costs)
            p[f] = best_loc
            used[best_loc] = True
            
        return p

    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    # Multi-start Tabu Search
    t0 = time.time()
    time_limit = 110.0
    
    best_p_overall = None
    best_cost_overall = np.iinfo(np.int64).max
    
    # We will run multiple independent Tabu Search instances if time permits,
    # or just one very strong one. Given 120s limit, one strong TS is safer.
    # Let's do 5 starts if time allows, otherwise 1.
    
    # Precompute for speed
    a_row_sum = a.sum(axis=1)
    
    # Start with the deterministic greedy solution
    p = greedy_construct(42)
    current_cost = total_cost(p)
    
    if current_cost < best_cost_overall:
        best_cost_overall = current_cost
        best_p_overall = p.copy()

    # Tabu Search Parameters
    tabu_len = 100  # Length of tabu list
    max_iter = 50000 # Max iterations per start
    rng = np.random.default_rng(123)
    
    # Tabu list stores (i, j) pairs of facility indices swapped
    # To avoid symmetry, we store (min(i,j), max(i,j))
    tabu_list = []
    
    # We will use a simple list for tabu, checking membership is O(L)
    # For n=100, L=100, this is fine.
    
    for start_iter in range(5): # Try up to 5 starts
        if time.time() - t0 > time_limit - 10:
            break
            
        if start_iter == 0:
            # Use the greedy solution
            pass
        else:
            # Perturb the best solution found so far
            p = best_p_overall.copy()
            # Apply 5 random swaps
            for _ in range(5):
                i = rng.integers(n)
                j = rng.integers(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = total_cost(p)
        
        # Initialize Tabu Search state
        current_p = p
        current_c = current_cost
        best_p_ts = current_p.copy()
        best_c_ts = current_c
        tabu_list = []
        
        # Precompute delta calculation vectors? 
        # We can't precompute for all pairs, but we can vectorize the delta calc.
        
        for it in range(max_iter):
            if time.time() - t0 > time_limit - 5:
                break
                
            # Select a move
            # We can scan all pairs (i, j) to find the best non-tabu move, 
            # or pick a random pair. Scanning all is O(n^2) per iter, which is 10000.
            # 50000 * 10000 = 500M ops. Might be too slow in Python loops.
            # Better: Pick random pair, calculate delta, check tabu.
            
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
                
            # Check Tabu
            key = (min(i, j), max(i, j))
            is_tabu = key in tabu_list
            
            # Calculate Delta
            u = current_p[i]
            v = current_p[j]
            
            # Delta calculation (same as seed)
            diff_row = a[i, :] - a[j, :]
            diff_col = a[:, i] - a[:, j]
            
            b_v_row = b[v, current_p]
            b_u_row = b[u, current_p]
            b_col_v = b[current_p, v]
            b_col_u = b[current_p, u]
            
            s1 = np.dot(diff_row, b_v_row - b_u_row)
            s2 = np.dot(diff_col, b_col_v - b_col_u)
            
            s1 -= diff_row[i] * (b[v, u] - b[u, u])
            s2 -= diff_col[i] * (b[u, v] - b[u, u])
            s1 -= diff_row[j] * (b[v, v] - b[u, v])
            s2 -= diff_col[j] * (b[v, v] - b[v, u])
            
            direct = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
            delta = s1 + s2 + direct
            
            new_cost = current_c + delta
            
            # Aspiration Criterion
            if not is_tabu or new_cost < best_c_ts:
                # Accept move
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_c = new_cost
                
                # Update Tabu List
                tabu_list.append(key)
                if len(tabu_list) > tabu_len:
                    tabu_list.pop(0)
                
                if new_cost < best_c_ts:
                    best_c_ts = new_cost
                    best_p_ts = current_p.copy()
                    
                    if new_cost < best_cost_overall:
                        best_cost_overall = new_cost
                        best_p_overall = current_p.copy()
                        
    return best_p_overall.tolist()
# EVOLVE-BLOCK-END
