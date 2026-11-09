"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(perm, pt):
    """Compute makespan for a permutation efficiently."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    if m == 1:
        return float(pt[perm, 0].sum())
    
    sub = pt[perm, :]
    comp = np.zeros((n, m))
    comp[0, 0] = sub[0, 0]
    for j in range(1, m):
        comp[0, j] = comp[0, j-1] + sub[0, j]
    for i in range(1, n):
        comp[i, 0] = comp[i-1, 0] + sub[i, 0]
        for j in range(1, m):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + sub[i, j]
    return float(comp[-1, -1])


def _neh_order(pt):
    """NEH heuristic: order jobs by total processing time descending, 
    inserting each job in the best position."""
    n = pt.shape[0]
    m = pt.shape[1]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [order[0]]
    
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm


def _palmer_key(pt):
    """Compute Palmer's key for each job."""
    n = pt.shape[0]
    m = pt.shape[1]
    keys = np.zeros(n)
    for j in range(n):
        for k in range(m):
            keys[j] += (n - 2*k - 1) * pt[j, k]
    return keys


def _local_search_block(perm, pt, time_limit=10.0):
    """Local search with Or-opt (block) moves and 2-opt swaps."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try 2-opt swaps
        for i in range(n - 1):
            if (time.time() - start_time) >= time_limit:
                break
            new_perm = best_perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms - 1e-9:
                best_perm = new_perm
                best_ms = ms
                improved = True
        
        # Try Or-opt: move a block of 2 or 3 jobs to a new position
        for block_size in [2, 3]:
            for i in range(n - block_size):
                if (time.time() - start_time) >= time_limit:
                    break
                block = best_perm[i:i+block_size]
                remaining = best_perm[:i] + best_perm[i+block_size:]
                for pos in range(len(remaining) + 1):
                    if pos == i:
                        continue
                    new_perm = remaining[:pos] + block + remaining[pos:]
                    ms = _makespan_fast(new_perm, pt)
                    if ms < best_ms - 1e-9:
                        best_perm = new_perm
                        best_ms = ms
                        improved = True
                        break
    
    return best_perm


def _aco_construct(pheromone, pt, rng, alpha=1.0, beta=2.0):
    """Construct a solution using ACO probabilistic rule with greedy insertion."""
    n = pt.shape[0]
    m = pt.shape[1]
    
    # Compute heuristic values (inverse of total processing time)
    totals = pt.sum(axis=1)
    heuristic = 1.0 / (totals + 1e-9)
    
    # Start with a random job
    remaining = list(range(n))
    perm = []
    
    # Pick first job using probabilistic rule
    probs = np.zeros(n)
    for j in remaining:
        probs[j] = (pheromone[j] ** alpha) * (heuristic[j] ** beta)
    total_prob = probs.sum()
    if total_prob == 0:
        perm.append(rng.integers(0, n))
        remaining.remove(perm[0])
    else:
        probs /= total_prob
        chosen = rng.choice(n, p=probs)
        perm.append(chosen)
        remaining.remove(chosen)
    
    # Insert remaining jobs one by one using greedy best insertion
    for _ in range(n - 1):
        if not remaining:
            break
        # Pick next job probabilistically
        probs = np.zeros(len(remaining))
        for idx, j in enumerate(remaining):
            probs[idx] = (pheromone[j] ** alpha) * (heuristic[j] ** beta)
        total_prob = probs.sum()
        if total_prob == 0:
            job = remaining[rng.integers(0, len(remaining))]
        else:
            probs /= total_prob
            chosen_idx = rng.choice(len(remaining), p=probs)
            job = remaining[chosen_idx]
        
        # Insert at best position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
        remaining.remove(job)
    
    return perm


def _ant_colony_optimization(pt, time_limit=30.0, n_ants=5, max_iter=50):
    """Ant Colony Optimization for flow shop scheduling."""
    n = pt.shape[0]
    m = pt.shape[1]
    start_time = time.time()
    
    # Initialize pheromone trails
    tau = np.ones(n)
    tau_min = 1e-6
    tau_max = 10.0
    rho = 0.5  # evaporation rate
    Q = 1.0
    
    # Initial solutions using heuristics
    neh_perm = _neh_order(pt)
    palmer_keys = _palmer_key(pt)
    palmer_perm = list(np.argsort(palmer_keys))
    
    # Evaluate initial solutions
    neh_ms = _makespan_fast(neh_perm, pt)
    palmer_ms = _makespan_fast(palmer_perm, pt)
    
    best_perm = neh_perm if neh_ms <= palmer_ms else palmer_perm
    best_ms = min(neh_ms, palmer_ms)
    
    # Initialize pheromone with best solution
    for pos, job in enumerate(best_perm):
        tau[job] += Q / best_ms
    
    rng = np.random.default_rng(42)
    iteration = 0
    
    while iteration < max_iter and (time.time() - start_time) < time_limit:
        iteration += 1
        
        # Construct solutions with ants
        for ant in range(n_ants):
            if (time.time() - start_time) >= time_limit:
                break
            
            # Construct solution
            perm = _aco_construct(tau, pt, rng, alpha=1.0, beta=2.0)
            ms = _makespan_fast(perm, pt)
            
            # Update best
            if ms < best_ms:
                best_ms = ms
                best_perm = list(perm)
        
        # Update pheromone trails
        # Evaporate
        tau *= (1 - rho)
        
        # Deposit pheromone on best solution found so far
        for job in best_perm:
            tau[job] += Q / max(best_ms, 1e-9)
        
        # Clip pheromone values
        tau = np.clip(tau, tau_min, tau_max)
        
        # Occasionally restart with a fresh heuristic solution to escape local optima
        if iteration % 10 == 0:
            alt_perm = list(rng.permutation(n))
            alt_ms = _makespan_fast(alt_perm, pt)
            if alt_ms < best_ms:
                best_ms = alt_ms
                best_perm = alt_perm
                for job in best_perm:
                    tau[job] += Q / max(best_ms, 1e-9)
    
    return best_perm, best_ms


def _simulated_annealing(perm, pt, time_limit=5.0):
    """Simulated annealing with swap moves."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    start_time = time.time()
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    T = max(pt.sum() / 10, 1.0)
    T_min = 1e-6
    cooling = 0.95
    
    rng = np.random.default_rng(42)
    
    while T > T_min and (time.time() - start_time) < time_limit:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _makespan_fast(new_perm, pt)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        T *= cooling
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 1:
        return list(np.argsort(pt[:, 0]))
    
    deadline = 140.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # Generate initial candidates
    candidates = []
    
    # 1. NEH heuristic
    neh_perm = _neh_order(pt)
    candidates.append(neh_perm)
    
    # 2. Palmer's key
    palmer_keys = _palmer_key(pt)
    palmer_perm = list(np.argsort(palmer_keys))
    candidates.append(palmer_perm)
    
    # 3. Longest total processing time
    lpt_perm = list(np.argsort(-pt.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 4. Shortest total processing time
    spt_perm = list(np.argsort(pt.sum(axis=1)))
    candidates.append(spt_perm)
    
    # 5. A few random permutations
    rng = np.random.default_rng(123)
    for _ in range(3):
        rand_perm = list(rng.permutation(n))
        candidates.append(rand_perm)
    
    # Evaluate all candidates and pick the best
    best_perm = candidates[0]
    best_ms = _makespan_fast(best_perm, pt)
    
    for perm in candidates[1:]:
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Local search on the best candidate
    remaining = deadline - 20.0 - (time.time() - start_time)
    if remaining > 3.0:
        ls_budget = min(10.0, remaining)
        best_perm = _local_search_block(best_perm, pt, time_limit=ls_budget)
        best_ms = _makespan_fast(best_perm, pt)
    
    # Ant Colony Optimization
    remaining = deadline - 5.0 - (time.time() - start_time)
    if remaining > 10.0:
        aco_budget = min(40.0, remaining)
        aco_perm, aco_ms = _ant_colony_optimization(pt, time_limit=aco_budget)
        if aco_ms < best_ms:
            best_perm = aco_perm
            best_ms = aco_ms
    
    # Final local search if time permits
    remaining = deadline - 3.0 - (time.time() - start_time)
    if remaining > 2.0:
        final_ls_budget = min(5.0, remaining)
        best_perm = _local_search_block(best_perm, pt, time_limit=final_ls_budget)
        best_ms = _makespan_fast(best_perm, pt)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
