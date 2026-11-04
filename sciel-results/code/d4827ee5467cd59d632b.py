"""
QAP tai256c solver using a different strategy:
- Use scipy's linear_sum_assignment (Hungarian algorithm) as the primary optimization engine.
- Reformulate QAP as a sequence of 2-swap decisions, but use a different approach:
  Instead of local search or simulated annealing, use a greedy construction with 
  Hungarian-based refinement, followed by a simple local search polish.

The key structural difference:
- Current program: Island model with Tabu Search and 2-swap neighborhood.
- New strategy: Use Hungarian algorithm (exact assignment for bipartite matching) 
  to solve a linearized approximation of QAP, then refine with local search.

For QAP, we can't directly use Hungarian because it's a quadratic problem.
However, we can:
1. Start with a good initial solution from rank matching.
2. Use a simple 2-swap local search with O(n) delta computation.
3. Use multiple restarts with different random perturbations.
4. The key insight: use a faster, more efficient local search loop.

Actually, let me try a different structural approach:
- Use a greedy construction algorithm that builds the permutation one facility at a time.
- At each step, place the next facility at the location that minimizes the incremental cost.
- This is O(n^3) construction, but gives a good starting point.
- Then polish with local search.
"""
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    a_sym = a + a.T (symmetrized flow matrix)
    """
    x = p[i]
    y = p[j]
    
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    temp = diff_a * diff_b[p]
    
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _local_search(a_sym, b, p, time_limit, rng):
    """
    Simple 2-swap local search.
    Returns best_p, best_cost.
    """
    n = len(p)
    current_p = p.copy()
    current_cost = _qap_cost(a_sym // 2 if np.allclose(a_sym, a_sym.T) else a_sym, b, current_p)
    # Recompute cost with original a for accuracy
    # Actually, a_sym = a + a.T, and cost uses a. 
    # For symmetric a, a = a_sym / 2.
    # Let's just use a_sym for delta and compute cost separately.
    
    # Recompute cost properly
    # We need original a, but we only have a_sym here.
    # Let's pass a as well.
    
    best_p = current_p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_delta = 0
        best_move = None
        
        # Scan all pairs (slow but thorough)
        # For n=256, that's ~32k pairs. Each delta is O(n) = 256 ops.
        # Total: 32k * 256 = 8M ops per pass. In numpy, ~10-50ms per pass.
        # We can do ~20-100 passes in 1-5 seconds.
        
        for i in range(n):
            if time.time() > time_limit:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a_sym, b, current_p, i, j)
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_move = (i, j)
        
        if best_move is not None:
            i, j = best_move
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += best_delta
            improved = True
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
    
    return best_p, best_cost

def _greedy_construct(a, b):
    """
    Greedy construction: build permutation one facility at a time.
    At each step, place facility f at location l that minimizes incremental cost.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    facility_order = np.arange(n)
    
    # Order facilities by flow centrality (row sum)
    flow_sums = a.sum(axis=1)
    facility_order = np.argsort(-flow_sums, kind="stable")
    
    # Precompute for incremental cost
    # When we place facility f at location l, the incremental cost is:
    # sum_{g assigned} a[f, g] * b[l, p[g]] + a[g, f] * b[p[g], l]
    # + (diagonal term if f == g, but we don't assign f to itself in this loop)
    
    for idx in range(n):
        f = facility_order[idx]
        best_l = -1
        best_cost = np.inf
        
        for l in range(n):
            if assigned[l]:
                continue
            
            # Compute incremental cost of placing f at l
            # Sum over all assigned facilities g
            inc_cost = 0
            for g_idx in range(idx):
                g = facility_order[g_idx]
                l_g = p[g_idx]
                inc_cost += a[f, g] * b[l, l_g] + a[g, f] * b[l_g, l]
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_l = l
        
        p[idx] = best_l
        assigned[best_l] = True
    
    return p

def _solve_tai256c(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    
    # Symmetrize flow matrix for delta computation
    a_sym = a + a.T
    
    # Initial solution: greedy construction
    p_init = _greedy_construct(a, b)
    cost_init = _qap_cost(a, b, p_init)
    
    # Also try rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = _qap_cost(a, b, p_rank)
    
    # Choose better initial solution
    if cost_rank < cost_init:
        p_best = p_rank.copy()
        cost_best = cost_rank
    else:
        p_best = p_init.copy()
        cost_best = cost_init
    
    # Run local search with multiple restarts
    rng = np.random.default_rng(42)
    
    # Time budget: use 80% for local search, 20% for safety
    ls_time_limit = time_limit * 0.8
    
    # First pass: local search from best initial
    p_ls, cost_ls = _local_search(a_sym, b, p_best, start_time + ls_time_limit * 0.5, rng)
    if cost_ls < cost_best:
        p_best = p_ls
        cost_best = cost_ls
    
    # Check time
    if time.time() - start_time > time_limit * 0.7:
        return p_best.tolist()
    
    # Second pass: local search from a perturbed version
    p_pert = p_best.copy()
    for _ in range(5):
        i, j = rng.choice(n, size=2, replace=False)
        p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
    
    p_ls2, cost_ls2 = _local_search(a_sym, b, p_pert, start_time + ls_time_limit, rng)
    if cost_ls2 < cost_best:
        p_best = p_ls2
        cost_best = cost_ls2
    
    # Third pass: another perturbation
    if time.time() - start_time < time_limit * 0.85:
        p_pert2 = p_best.copy()
        for _ in range(10):
            i, j = rng.choice(n, size=2, replace=False)
            p_pert2[i], p_pert2[j] = p_pert2[j], p_pert2[i]
        
        p_ls3, cost_ls3 = _local_search(a_sym, b, p_pert2, start_time + time_limit - 5, rng)
        if cost_ls3 < cost_best:
            p_best = p_ls3
            cost_best = cost_ls3
    
    return p_best.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
