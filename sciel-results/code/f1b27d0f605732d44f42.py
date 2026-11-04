"""Rewritten QAP solver using Tabu Search with efficient delta evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    B_p = b[np.ix_(p, p)]
    return int(np.sum(a * B_p))

def _initial_solution(a, b):
    """
    Generate an initial solution using rank matching heuristic.
    Facilities are sorted by total flow (descending), locations by total distance (ascending).
    """
    n = len(a)
    # Sum of flows for each facility
    flow_sums = a.sum(axis=1)
    # Sum of distances for each location
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by flow (heaviest first)
    heavy = np.argsort(-flow_sums, kind="stable")
    # Sort locations by distance (closest to center/lowest sum first)
    central = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    return p

def _two_opt_local_search(a, b, p, max_passes=10):
    """
    Perform 2-opt local search until no improvement is found.
    Uses O(n) delta calculation for each swap.
    """
    n = len(a)
    current_cost = _compute_cost(a, b, p)
    
    for _ in range(max_passes):
        improved = False
        # Iterate over all pairs (i, j)
        for i in range(n):
            u = p[i]
            for j in range(i + 1, n):
                v = p[j]
                
                # Calculate delta cost for swapping i and j
                # Delta = sum_{k != i,j} [ (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) 
                #                         + (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u]) ]
                #       + (a[i,j] - a[j,i]) * (b[v, u] - b[u, v])
                
                # Vectorized calculation for k != i, j
                # Create mask for k != i, j
                # To avoid masking overhead, we can include i and j and subtract their contributions later,
                # but the formula is derived for k != i, j.
                # Let's compute the sum over all k, then subtract k=i and k=j terms.
                
                # Term 1: sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
                # b[v, p] is a vector. b[u, p] is a vector.
                diff_b_row = b[v, p] - b[u, p]
                diff_a_row = a[i, :] - a[j, :]
                sum1_all = np.dot(diff_a_row, diff_b_row)
                
                # Subtract k=i term: (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
                # Note: p[i] = u, so p[k] for k=i is u.
                term_k_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                
                # Subtract k=j term: (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
                # Note: p[j] = v, so p[k] for k=j is v.
                term_k_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                
                sum1 = sum1_all - term_k_i - term_k_j
                
                # Term 2: sum_k (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
                # b[p, v] is a vector (column v of b indexed by p).
                # b[p, u] is a vector (column u of b indexed by p).
                diff_b_col = b[p, v] - b[p, u]
                diff_a_col = a[:, i] - a[:, j]
                sum2_all = np.dot(diff_a_col, diff_b_col)
                
                # Subtract k=i term: (a[i,i] - a[i,j]) * (b[u, v] - b[u, u])
                term_k_i_2 = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                
                # Subtract k=j term: (a[j,i] - a[j,j]) * (b[v, v] - b[v, u])
                term_k_j_2 = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                
                sum2 = sum2_all - term_k_i_2 - term_k_j_2
                
                # Term 3: Direct interaction
                term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                
                delta = sum1 + sum2 + term3
                
                if delta < -1e-9: # If improvement
                    # Perform swap
                    p[i], p[j] = v, u
                    current_cost += delta
                    improved = True
        if not improved:
            break
            
    return p, current_cost

def _tabu_search(a, b, p, time_limit=140.0):
    """
    Perform Tabu Search.
    """
    n = len(a)
    start_time = time.time()
    current_cost = _compute_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu list size
    tabu_size = 20
    tabu_list = [] # List of (i, j) swaps, or just store last moves
    
    # Precompute some static parts?
    # The delta calculation depends on p, so we can't precompute much.
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_i = -1
        best_j = -1
        best_is_tabu = False
        
        # Evaluate all moves
        for i in range(n):
            u = p[i]
            for j in range(i + 1, n):
                v = p[j]
                
                # Check if move is tabu
                # Simple check: if (i,j) or (j,i) was in last tabu_size moves
                # To make this O(1), we could use a counter array, but for n=150, 
                # checking a small list is fine.
                is_tabu = False
                # Optimization: Store tabu moves in a set for O(1) lookup
                # But we need to expire them.
                # Let's use a list of (time_step, i, j) and filter.
                # Or simpler: a dictionary mapping (i,j) to expiration time.
                # For now, simple list check.
                # Actually, let's just compute delta first, then check tabu.
                
                # Calculate delta (same as in 2-opt)
                diff_b_row = b[v, p] - b[u, p]
                diff_a_row = a[i, :] - a[j, :]
                sum1_all = np.dot(diff_a_row, diff_b_row)
                term_k_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                term_k_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                sum1 = sum1_all - term_k_i - term_k_j
                
                diff_b_col = b[p, v] - b[p, u]
                diff_a_col = a[:, i] - a[:, j]
                sum2_all = np.dot(diff_a_col, diff_b_col)
                term_k_i_2 = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                term_k_j_2 = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                sum2 = sum2_all - term_k_i_2 - term_k_j_2
                
                term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                delta = sum1 + sum2 + term3
                
                # Check tabu status
                # We need to know if this move is in the tabu list.
                # Let's maintain a separate structure for this in the main loop.
                # For now, assume we check it later.
                
                # Aspiration criterion: if delta < best_cost - current_cost (i.e., new global best), accept even if tabu.
                # Otherwise, if not tabu and delta < best_delta, accept.
                
                # We need to know if it's tabu.
                # Let's pass tabu_list and check.
                # To avoid O(T) check per move, we can use a set.
                
                # Let's restructure to handle tabu properly.
                
        # The above loop is O(n^3) per iteration.
        # 150^3 = 3.3M. 
        # 140 seconds / (time for 3.3M ops).
        # If 3.3M ops takes 1s, we can do 140 iterations.
        
        # This is a bit slow in pure Python loops.
        # I will optimize the inner loop by using NumPy vectorization for the delta calculation 
        # for a fixed i against all j.
        
        # Let's break out and implement a more vectorized version.
        break 
        
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Initial Solution
    p = _initial_solution(a, b)
    
    # 2. Local Search to polish initial solution
    p, current_cost = _two_opt_local_search(a, b, p)
    
    # 3. Tabu Search
    # We will implement a simpler, faster tabu search loop here.
    # To handle the O(n^3) neighborhood evaluation efficiently, we will iterate i, 
    # and for each i, compute deltas for all j > i using vectorized operations where possible.
    
    best_p = p.copy()
    best_cost = current_cost
    
    tabu_size = 15
    tabu_expiration = {} # key: (i,j), value: expiration_step
    
    step = 0
    while time.time() < deadline:
        step += 1
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            u = p[i]
            # Vectorized delta calculation for all j > i
            # This is complex to fully vectorize due to p[j] dependency in b[v, p[k]].
            # We will stick to the loop but optimize the Python overhead.
            
            for j in range(i + 1, n):
                v = p[j]
                
                # Tabu check
                key = (i, j)
                if key in tabu_expiration and tabu_expiration[key] > step:
                    is_tabu = True
                else:
                    is_tabu = False
                
                # Calculate Delta
                diff_b_row = b[v, p] - b[u, p]
                diff_a_row = a[i, :] - a[j, :]
                sum1_all = np.dot(diff_a_row, diff_b_row)
                term_k_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                term_k_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                sum1 = sum1_all - term_k_i - term_k_j
                
                diff_b_col = b[p, v] - b[p, u]
                diff_a_col = a[:, i] - a[:, j]
                sum2_all = np.dot(diff_a_col, diff_b_col)
                term_k_i_2 = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                term_k_j_2 = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                sum2 = sum2_all - term_k_i_2 - term_k_j_2
                
                term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                delta = sum1 + sum2 + term3
                
                # Acceptance criteria
                # 1. If delta < best_delta and (not tabu or aspiration)
                # Aspiration: new solution cost < best_cost
                new_cost = current_cost + delta
                if new_cost < best_cost:
                    aspiration = True
                else:
                    aspiration = False
                
                if delta < best_delta and (not is_tabu or aspiration):
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        # If no move found (should not happen), break
        if best_i == -1:
            break
            
        # Perform the move
        i, j = best_i, best_j
        u, v = p[i], p[j]
        p[i], p[j] = v, u
        current_cost += best_delta
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update Tabu List
        tabu_expiration[(i, j)] = step + tabu_size
        tabu_expiration[(j, i)] = step + tabu_size
        
        # Clean up old tabu entries (optional, for memory)
        if step % 100 == 0:
            for k in list(tabu_expiration.keys()):
                if tabu_expiration[k] <= step:
                    del tabu_expiration[k]
                    
    return best_p.tolist()
# EVOLVE-BLOCK-END
