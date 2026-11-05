# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        pt_row = pt[job]
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            comp[mi] = max(comp[mi], prev) + pt_row[mi]
    return comp[-1]

def _makespan_fast(perm, pt, comp_cache=None):
    """Compute makespan, optionally using a pre-allocated cache."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    if comp_cache is None:
        comp = np.zeros(m, dtype=float)
    else:
        comp = comp_cache
        comp[:] = 0.0
    for j in range(n):
        job = perm[j]
        pt_row = pt[job]
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            comp[mi] = max(comp[mi], prev) + pt_row[mi]
    return comp[-1]

def _neh_initial(pt):
    """NEH heuristic: build permutation greedily by inserting each job (sorted by decreasing total) at the best position."""
    n, m = pt.shape
    if n <= 1:
        return np.arange(n)
    
    total_pt = pt.sum(axis=1)
    order = np.argsort(-total_pt)
    
    perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(trial, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=int)

def _palmer_initial(pt):
    """Palmer's heuristic."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    keys = np.zeros(n)
    for j in range(n):
        for mi in range(m):
            keys[j] += (n - 1 - 2 * mi) * pt[j, mi]
    return np.argsort(-keys).astype(int)

def _vnd_search(perm, pt, time_limit, start_time):
    """
    Variable Neighborhood Descent: systematically cycles through nested neighborhoods
    of increasing radius. Neighborhoods:
      N1: adjacent swaps (radius 1)
      N2: 2-opt reversals (radius 2)
      N3: block insertions of size 1-3 (radius 3)
    Accepts first improvement found in each neighborhood before expanding.
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    comp_cache = np.zeros(pt.shape[1], dtype=float)
    
    # Neighborhood structures
    def _neighborhood_adjacent(perm, ms):
        """N1: adjacent swaps."""
        for i in range(len(perm) - 1):
            new_perm = perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_ms = _makespan_fast(new_perm, pt, comp_cache)
            if new_ms < ms:
                return new_perm, new_ms
        return None, ms
    
    def _neighborhood_2opt(perm, ms):
        """N2: 2-opt reversals (reverse segment of length >= 2)."""
        for i in range(len(perm)):
            for j in range(i + 2, len(perm)):
                new_perm = perm.copy()
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                new_ms = _makespan_fast(new_perm, pt, comp_cache)
                if new_ms < ms:
                    return new_perm, new_ms
        return None, ms
    
    def _neighborhood_block_insert(perm, ms):
        """N3: move a block of size 1-3 to a different position."""
        for block_size in [1, 2, 3]:
            for i in range(len(perm) - block_size + 1):
                block = perm[i:i+block_size]
                remaining = np.concatenate([perm[:i], perm[i+block_size:]])
                for j in range(len(remaining) + 1):
                    if j == i:
                        continue
                    new_perm = np.insert(remaining, j, block)
                    new_ms = _makespan_fast(new_perm, pt, comp_cache)
                    if new_ms < ms:
                        return new_perm, new_ms
        return None, ms
    
    neighborhoods = [_neighborhood_adjacent, _neighborhood_2opt, _neighborhood_block_insert]
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for neighborhood in neighborhoods:
            if time.time() - start_time >= time_limit:
                break
            new_perm, new_ms = neighborhood(best_perm, best_ms)
            if new_perm is not None:
                best_perm = new_perm
                best_ms = new_ms
                improved = True
                break  # Restart from N1 after any improvement
    
    return best_perm, best_ms

def _iterated_vnd(pt, start_time, deadline, num_iters=20, perturb_size=3):
    """
    Iterated VND: repeatedly apply VND, then perturb the solution to escape local optima.
    Perturbation: random block insertion of size 2-4.
    """
    n = pt.shape[0]
    if n <= 1:
        return pt, _makespan(pt, pt) if n > 0 else 0.0
    
    best_perm = pt.copy()
    best_ms = _makespan(best_perm, pt)
    comp_cache = np.zeros(pt.shape[1], dtype=float)
    
    for it in range(num_iters):
        if time.time() - start_time >= deadline:
            break
        
        # Apply VND
        curr_perm, curr_ms = _vnd_search(best_perm, pt, 5.0, start_time)
        if curr_ms < best_ms:
            best_ms = curr_ms
            best_perm = curr_perm.copy()
        
        # Perturb: move a random block of size 2-4
        if n < 4:
            continue
        
        block_size = min(np.random.randint(2, 5), n - 1)
        i = np.random.randint(0, n - block_size + 1)
        block = best_perm[i:i+block_size]
        remaining = np.concatenate([best_perm[:i], best_perm[i+block_size:]])
        j = np.random.randint(0, len(remaining) + 1)
        if j == i:
            j = (j + 1) % (len(remaining) + 1)
        perturbed = np.insert(remaining, j, block)
        
        # Apply VND to perturbed solution
        curr_perm, curr_ms = _vnd_search(perturbed, pt, 5.0, start_time)
        if curr_ms < best_ms:
            best_ms = curr_ms
            best_perm = curr_perm.copy()
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0
    effective_deadline = deadline - safety_margin
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # NEH (greedy insertion)
    perm_neh = _neh_initial(processing_times)
    initial_solutions.append(perm_neh)
    
    # Palmer
    perm_palmer = _palmer_initial(processing_times)
    initial_solutions.append(perm_palmer)
    
    # Simple NEH (sort by total)
    total_pt = processing_times.sum(axis=1)
    perm_simple_neh = np.argsort(-total_pt).astype(int)
    initial_solutions.append(perm_simple_neh)
    
    # Bottleneck
    machine_totals = processing_times.sum(axis=0)
    bottleneck_machine = np.argmax(machine_totals)
    perm_bn = np.argsort(-processing_times[:, bottleneck_machine]).astype(int)
    initial_solutions.append(perm_bn)
    
    # First machine
    perm_first = np.argsort(-processing_times[:, 0]).astype(int)
    initial_solutions.append(perm_first)
    
    # Last machine
    perm_last = np.argsort(-processing_times[:, -1]).astype(int)
    initial_solutions.append(perm_last)
    
    # Random solutions
    for _ in range(6):
        perm_rand = rng.permutation(n)
        initial_solutions.append(perm_rand)
    
    # Evaluate all initial solutions
    best_perm = None
    best_ms = float('inf')
    comp_cache = np.zeros(m, dtype=float)
    
    for perm in initial_solutions:
        ms = _makespan_fast(perm, processing_times, comp_cache)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
    
    # Phase 1: VND from best initial solution
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _vnd_search(best_perm, processing_times, 15.0, start_time)
    
    # Phase 2: Iterated VND with perturbations
    if time.time() - start_time < effective_deadline:
        ivnd_perm, ivnd_ms = _iterated_vnd(best_perm, start_time, effective_deadline, num_iters=15, perturb_size=3)
        if ivnd_ms < best_ms:
            best_ms = ivnd_ms
            best_perm = ivnd_perm.copy()
    
    # Phase 3: Run VND from other good initial solutions
    if time.time() - start_time < effective_deadline:
        for perm in initial_solutions[:6]:
            if time.time() - start_time >= effective_deadline:
                break
            curr_perm, curr_ms = _vnd_search(perm, processing_times, 8.0, start_time)
            if curr_ms < best_ms:
                best_ms = curr_ms
                best_perm = curr_perm.copy()
    
    # Phase 4: Final polish with VND
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _vnd_search(best_perm, processing_times, 10.0, start_time)
    
    # Phase 5: Random restarts with VND if time permits
    remaining_time = effective_deadline - (time.time() - start_time)
    if remaining_time > 10.0:
        num_restarts = min(5, int(remaining_time // 15))
        for _ in range(num_restarts):
            if time.time() - start_time >= effective_deadline:
                break
            perm_rand = rng.permutation(n)
            curr_perm, curr_ms = _vnd_search(perm_rand, processing_times, 12.0, start_time)
            if curr_ms < best_ms:
                best_ms = curr_ms
                best_perm = curr_perm.copy()
    
    return list(best_perm)
# EVOLVE-BLOCK-END
