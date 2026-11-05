"""Flow-shop permutation via multi-start heuristics + fast insertion local search + segment reversal."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """
    Fast makespan calculation using pure Python lists.
    pt_list: list of lists, shape [n_jobs][n_machines].
    perm: list of job indices.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0:
        return 0.0
    if n_machines == 0:
        return 0.0

    prev = [0.0] * n_machines

    for job in perm:
        row = pt_list[job]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]


def _build_insertion_costs(pt_list, n_jobs):
    """
    Pre-compute an N x N matrix where cost[i][j] approximates the marginal
    makespan increase when job i is placed immediately before job j.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    costs = [[0.0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                continue
            row_i = pt_list[i]
            row_j = pt_list[j]
            c = row_i[0]
            for m in range(1, n_machines):
                diff = row_i[m] - row_j[m]
                if diff > 0:
                    c += diff
            costs[i][j] = c
    return costs


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_list = pt.tolist()
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's dispatching rule."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _spt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _bottleneck_initial(pt):
    """Sort by total processing on the machine with largest total load, descending."""
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _mean_initial(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first_machine_initial(pt):
    return list(np.argsort(-pt[:, 0]))


def _last_machine_initial(pt):
    return list(np.argsort(-pt[:, -1]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _vnd_adjacent_swap(pt_list, perm, deadline):
    """N1: Adjacent swap neighborhood."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            test = current[:i] + [current[i+1]] + [current[i]] + current[i+2:]
            ms = _makespan_fast(pt_list, test)
            if ms < current_ms - 1e-9:
                current = test
                current_ms = ms
                improved = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
                break
    return best, best_ms


def _vnd_2opt(pt_list, perm, deadline):
    """N2: 2-opt segment reversal neighborhood."""
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test = current[:i] + current[i:j+1][::-1] + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current = test
                    current_ms = ms
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
    return best, best_ms


def _vnd_or_opt(pt_list, perm, deadline):
    """N3: Or-opt block insertion neighborhood (blocks of 1-3)."""
    n = len(perm)
    if n <= 3:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for block_size in [1, 2, 3]:
            if time.time() > deadline:
                break
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    break
                block = current[i:i+block_size]
                rest = current[:i] + current[i+block_size:]
                
                for j in range(len(rest) + 1):
                    if time.time() > deadline:
                        break
                    if j == i:
                        continue
                    if j == i + 1:
                        continue
                    test = rest[:j] + block + rest[j:]
                    if len(test) != n:
                        continue
                    ms = _makespan_fast(pt_list, test)
                    if ms < current_ms - 1e-9:
                        current = test
                        current_ms = ms
                        improved = True
                        if current_ms < best_ms - 1e-9:
                            best_ms = current_ms
                            best = list(current)
                        break
                if improved:
                    break
            if improved:
                break
    return best, best_ms


def _vnd_local_search(pt_list, perm, deadline, max_cycles=20):
    """
    Variable Neighborhood Descent: systematically cycles through nested neighborhoods
    of increasing radius, accepting first improvement in each before expanding.
    """
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    neighborhoods = [
        _vnd_adjacent_swap,
        _vnd_2opt,
        _vnd_or_opt,
    ]
    
    for cycle in range(max_cycles):
        if time.time() > deadline:
            break
        improved_in_cycle = False
        for neighborhood in neighborhoods:
            if time.time() > deadline:
                break
            perm_new, ms_new = neighborhood(pt_list, current, deadline)
            if ms_new < current_ms - 1e-9:
                current = perm_new
                current_ms = ms_new
                improved_in_cycle = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
        if not improved_in_cycle:
            break
    
    return best, best_ms


def _local_search_2opt(pt_list, perm, deadline, max_rounds=5000, cost_matrix=None):
    """
    Local search using 2-opt segment reversal: reverse the segment from i to j.
    """
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if cost_matrix is not None and (j - i) > 2:
                    if i > 0:
                        c_cur = cost_matrix[current[i]][current[i-1]]
                        c_rev = cost_matrix[current[j]][current[i-1]]
                        if c_rev > c_cur * 1.5:
                            continue
                    if j < n - 1:
                        c_cur = cost_matrix[current[j]][current[j+1]]
                        c_rev = cost_matrix[current[i]][current[j+1]]
                        if c_rev > c_cur * 1.5:
                            continue
                test = current[:i] + current[i:j+1][::-1] + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _or_opt_local_search(pt_list, perm, deadline, max_rounds=5000, cost_matrix=None):
    """
    Or-opt local search: remove a block of 1-3 consecutive jobs and re-insert elsewhere.
    This escapes local optima that single-job insertion cannot.
    """
    n = len(perm)
    if n <= 3:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        for block_size in [1, 2, 3]:
            if time.time() > deadline:
                break
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    break
                block = current[i:i+block_size]
                rest = current[:i] + current[i+block_size:]
                
                for j in range(len(rest) + 1):
                    if time.time() > deadline:
                        break
                    # Skip trivial moves
                    if j == i:
                        continue
                    if j == i + 1 and block_size == 1:
                        continue
                    if j == i + 1 and block_size > 1:
                        # Check if it's the same position
                        if j == i + 1:
                            continue
                            
                    test = rest[:j] + block + rest[j:]
                    if len(test) != n:
                        continue
                    ms = _makespan_fast(pt_list, test)
                    if ms < current_ms - 1e-9:
                        current_ms = ms
                        current = test
                        improved = True
                        if current_ms < best_ms - 1e-9:
                            best_ms = current_ms
                            best = list(current)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    return best, best_ms


def _simulated_annealing(pt_list, perm, deadline, max_iter=100000, seed=0, cost_matrix=None):
    """Simulated annealing with insertion moves, biased by cost matrix if available."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    cooling = 0.9995

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        if cost_matrix is not None:
            j = rng.randint(0, n - 1)
            if rng.random() < 0.3:
                best_j = 0
                best_cost = float('inf')
                for jj in range(n):
                    if jj == i or jj == i + 1:
                        continue
                    pred = current[jj-1] if jj > 0 else current[0]
                    c = cost_matrix[job][pred]
                    if c < best_cost:
                        best_cost = c
                        best_j = jj
                if rng.random() < 0.5:
                    j = best_j
        else:
            j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(pt_list, test)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = test
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms


def _gravitational_search(pt_list, perm, deadline, max_iter=5000, seed=42):
    """
    Gravitational Search Algorithm (GSA) adapted for permutation optimization.
    Jobs are masses; makespan is potential energy. Forces attract good permutations.
    Uses a velocity-based position update with permutation repair.
    """
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    # Population-based GSA
    pop_size = min(10, max(4, n // 2))
    population = [list(current)]
    # Add diverse initial solutions
    for _ in range(pop_size - 1):
        p = list(current)
        # Random perturbation: swap 2-3 random pairs
        for _ in range(rng.randint(1, 3)):
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1)
            p[a], p[b] = p[b], p[a]
        population.append(p)
    
    # Compute initial fitnesses
    fitnesses = [_makespan_fast(pt_list, p) for p in population]
    
    T = max(best_ms * 0.02, 1.0)
    T_min = max(best_ms * 0.001, 0.01)
    cooling = 0.999
    
    for _iter in range(max_iter):
        if time.time() > deadline:
            break
        
        # Update best
        for i, (p, f) in enumerate(zip(population, fitnesses)):
            if f < best_ms:
                best_ms = f
                best = list(p)
        
        # Compute masses (inverse of makespan, normalized)
        min_f = min(fitnesses)
        max_f = max(fitnesses)
        if max_f - min_f < 1e-9:
            masses = [1.0] * pop_size
        else:
            masses = [(max_f - f) / (max_f - min_f) + 0.1 for f in fitnesses]
        total_mass = sum(masses)
        masses = [m / total_mass for m in masses]
        
        # Compute gravitational forces and update positions
        new_population = []
        for i in range(pop_size):
            # Compute net force from all other particles
            force = [0.0] * n
            for j in range(pop_size):
                if i == j:
                    continue
                # Force direction: towards better solutions
                # Use position difference as proxy
                for pos in range(n):
                    if population[i][pos] != population[j][pos]:
                        # Attract towards better solution
                        strength = masses[j] * (fitnesses[i] - fitnesses[j]) / max(1.0, fitnesses[i])
                        if fitnesses[j] < fitnesses[i]:
                            force[pos] += strength
                        else:
                            force[pos] -= strength * 0.5
            
            # Generate new candidate by biased mutation based on forces
            new_perm = list(population[i])
            for pos in range(n):
                if abs(force[pos]) > 0.01 and rng.random() < 0.3:
                    # Swap with a random position
                    other = rng.randint(0, n - 1)
                    new_perm[pos], new_perm[other] = new_perm[other], new_perm[pos]
            
            # Also try Or-opt style moves
            if rng.random() < 0.2:
                block_size = rng.randint(1, 3)
                if n > block_size:
                    i_start = rng.randint(0, n - block_size)
                    block = new_perm[i_start:i_start + block_size]
                    rest = new_perm[:i_start] + new_perm[i_start + block_size:]
                    j_pos = rng.randint(0, len(rest))
                    new_perm = rest[:j_pos] + block + rest[j_pos:]
            
            new_population.append(new_perm)
        
        # Evaluate new population
        new_fitnesses = [_makespan_fast(pt_list, p) for p in new_population]
        
        # Selection: keep best half, replace worst half with new candidates
        combined = list(zip(fitnesses, population)) + list(zip(new_fitnesses, new_population))
        combined.sort(key=lambda x: x[0])
        population = [p for _, p in combined[:pop_size]]
        fitnesses = [f for f, _ in combined[:pop_size]]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()

    # Pre-compute insertion cost matrix for guided search
    cost_matrix = _build_insertion_costs(pt_list, n_jobs)

    # Build diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    # Add random starts
    for s in range(6):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, pick best
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # Sort candidates by makespan so we polish the best ones first
    candidate_results = []
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        candidate_results.append((ms, cand))
    candidate_results.sort(key=lambda x: x[0])

    # Multi-start local search: polish top candidates using VND
    for ms0, cand in candidate_results:
        if time.time() > deadline - 15.0:
            break
        # Skip if too far from best
        if ms0 > best_ms * 1.2:
            continue

        # VND local search: systematically explores nested neighborhoods
        perm, ms = _vnd_local_search(pt_list, cand, deadline - 15.0, max_cycles=10)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Gravitational search from best to explore broader space
    if time.time() < deadline - 20.0:
        perm_gsa, ms_gsa = _gravitational_search(pt_list, best_perm, deadline - 20.0, max_iter=3000, seed=7)
        if ms_gsa < best_ms:
            best_ms = ms_gsa
            best_perm = perm_gsa

    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 15.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 15.0, max_iter=200000, seed=7, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final VND polish
    if time.time() < deadline - 3.0:
        perm, ms = _vnd_local_search(pt_list, best_perm, deadline - 2.0, max_cycles=15)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final adjacent swap polish (fast, fine-tuning)
    if time.time() < deadline - 1.0:
        perm_swap, ms_swap = _vnd_adjacent_swap(pt_list, best_perm, deadline - 0.5)
        if ms_swap < best_ms:
            best_ms = ms_swap
            best_perm = perm_swap

    return best_perm
# EVOLVE-BLOCK-END
