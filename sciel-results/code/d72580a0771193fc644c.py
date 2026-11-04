"""
Memetic Genetic Algorithm (GA) for QAPLIB wil100 with optimized delta computation.
Strategy:
1. Population-based search with vectorized O(n^2) swap delta computation for fast local search.
2. Cycle Crossover + multi-point swap mutation.
3. Tournament selection with elitism.
4. Optimized time allocation: more generations with lighter local search, plus a final polish phase.
5. Integrated Simulated Annealing (SA) polish using efficient O(n) delta calculation for final refinement.
"""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b, mode='row'):
    """
    Generate an initial permutation using the Hungarian algorithm on a surrogate cost matrix.
    mode: 'row' or 'col'
    """
    n = len(a)
    if mode == 'row':
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        C = -np.outer(a_sums, b_sums)
    elif mode == 'col':
        a_sums = a.sum(axis=0)
        b_sums = b.sum(axis=0)
        C = -np.outer(a_sums, b_sums)
    else:
        C = -np.outer(a.sum(axis=1), b.sum(axis=1))
        
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _compute_all_swap_deltas(a, b, p):
    """
    Compute the cost delta for swapping every pair of positions (i, j) 
    in the current permutation p.
    
    Returns a matrix D where D[i, j] is the change in cost if we swap 
    positions i and j in p.
    """
    C = b[np.ix_(p, p)]
    # Precompute symmetric part of a for reuse
    a_sym = a + a.T
    
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    # Efficient computation using broadcasting
    Term1 = S1 + S1.T - D_S1[:, None] - D_S1[None, :]
    Term2 = T1 + T1.T - D_T1[:, None] - D_T1[None, :]
    
    Raw = Term1 + Term2
    
    # Error correction term
    Error = -2.0 * a_sym * C
    
    Delta = Raw - Error
    
    # Zero out diagonal
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _local_search(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search with early termination.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        deltas = _compute_all_swap_deltas(a, b, p)
        
        # Find the best improving move
        min_idx = np.argmin(deltas)
        min_delta = deltas.flat[min_idx]
        
        if min_delta >= 0:
            break # Local minimum reached
            
        # Find indices of the best move
        i, j = np.unravel_index(min_idx, deltas.shape)
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
    return p, current_cost

def _cycle_crossover(p1, p2):
    """
    Cycle Crossover (CX).
    """
    n = len(p1)
    child = np.full(n, -1, dtype=np.int64)
    
    visited = np.zeros(n, dtype=bool)
    
    # We need to map values to positions in p2 for O(1) lookup
    pos_in_p2 = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos_in_p2[p2[i]] = i
        
    # Standard CX algorithm
    s = random.randint(0, n - 1)
    current = s
    while not visited[current]:
        visited[current] = True
        child[current] = p1[current]
        val = p1[current]
        current = pos_in_p2[val]
        
    # Fill the rest from p2
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
            
    return child

def _mutate(p, num_swaps):
    """
    Apply random swaps to the permutation.
    """
    n = len(p)
    for _ in range(num_swaps):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def _sa_polish(a, b, p, time_limit):
    """
    Simulated Annealing polish phase.
    Uses efficient O(n) delta calculation for each swap.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Cooling schedule
    t0 = 8000.0
    t_final = 0.05
    
    batch_size = 50  # Process multiple swaps per time check
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        for _ in range(batch_size):
            if time.time() - start_time > time_limit:
                break
                
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            pi = p[i]
            pj = p[j]
            
            # Fast delta computation using vectorized dot products
            # Direct pair term
            delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
            
            # Diagonal terms
            delta += a[i, i] * (b[pj, pj] - b[pi, pi])
            delta += a[j, j] * (b[pi, pi] - b[pj, pj])
            
            # Cross terms using vectorized dot products
            diff_a_row = a[i, :] - a[j, :]
            diff_b_col = b[pj, p] - b[pi, p]
            sum_T1 = np.dot(diff_a_row, diff_b_col)
            sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
            sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
            
            diff_a_col = a[:, i] - a[:, j]
            diff_b_row = b[p, pj] - b[p, pi]
            sum_T2 = np.dot(diff_a_col, diff_b_row)
            sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
            sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
            
            delta += sum_T1 + sum_T2
            delta = int(delta)
            
            # Acceptance criterion
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # 5s buffer
    start_time = time.time()
    
    # Population parameters
    pop_size = 40
    elite_size = 4
    mutation_swaps = 2
    
    # Initialize population
    population = []
    costs = []
    
    # Add Hungarian inits with moderate local search
    for mode in ['row', 'col']:
        p = _hungarian_initial(a, b, mode)
        p, c = _local_search(a, b, p, max_iters=15, time_limit=2.0, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    # Fill rest with random + lighter local search
    while len(population) < pop_size:
        p = np.random.permutation(n)
        p, c = _local_search(a, b, p, max_iters=8, time_limit=1.0, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    best_idx = np.argmin(costs)
    best_p = population[best_idx].copy()
    best_cost = costs[best_idx]
    
    # Main GA Loop
    gen = 0
    # Reserve 15s for final SA polish
    while time.time() - start_time < total_deadline - 15.0:
        # Sort population by cost
        indices = np.argsort(costs)
        population = [population[i] for i in indices]
        costs = [costs[i] for i in indices]
        
        # Update global best
        if costs[0] < best_cost:
            best_cost = costs[0]
            best_p = population[0].copy()
            
        # Elitism: Keep the best 'elite_size'
        new_population = [population[i].copy() for i in range(elite_size)]
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < pop_size:
            # Tournament selection
            candidates = random.sample(range(len(population)), 3)
            best_cand = min(candidates, key=lambda x: costs[x])
            p1 = population[best_cand]
            
            candidates = random.sample(range(len(population)), 3)
            best_cand = min(candidates, key=lambda x: costs[x])
            p2 = population[best_cand]
            
            # Crossover
            child = _cycle_crossover(p1, p2)
            
            # Mutation
            child = _mutate(child, mutation_swaps)
            
            # Check time remaining
            time_left = total_deadline - (time.time() - start_time)
            if time_left < 1.0:
                break
            
            # Lighter local search
            child, c = _local_search(a, b, child, max_iters=12, time_limit=0.3, start_time=time.time())
            
            new_population.append(child)
            
        population = new_population
        costs = [_qap_cost(a, b, p) for p in population]
        
        gen += 1
    
    # Final polish phase: Intensive SA + Local Search
    time_left = total_deadline - (time.time() - start_time)
    
    # 1. Run SA for most of the remaining time
    if time_left > 10.0:
        sa_time = time_left - 5.0
        best_p, best_cost = _sa_polish(a, b, best_p, sa_time)
        
        # 2. Final local search to clean up
        time_left = total_deadline - (time.time() - start_time)
        if time_left > 2.0:
            best_p, best_cost = _local_search(a, b, best_p, max_iters=100, time_limit=time_left - 1.0, start_time=start_time)
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
