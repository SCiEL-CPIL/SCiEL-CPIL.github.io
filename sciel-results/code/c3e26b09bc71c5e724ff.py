"""Flow-shop permutation construction using NEH variants + GA with OX + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Use list-based approach for small n_machines to reduce NumPy overhead
    if n_machines <= 10:
        first_job = order[0]
        row = P[first_job]
        curr_row = [0.0] * n_machines
        curr_row[0] = row[0]
        for m in range(1, n_machines):
            curr_row[m] = curr_row[m-1] + row[m]
            
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new_row = [0.0] * n_machines
            new_row[0] = curr_row[0] + row[0]
            for m in range(1, n_machines):
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        return curr_row[-1]
    else:
        first_job = order[0]
        curr_row = np.cumsum(P[first_job])
        
        for idx in range(1, n_jobs):
            j = order[idx]
            prev_row = curr_row
            new_row = np.empty(n_machines)
            new_row[0] = prev_row[0] + P[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
            curr_row = new_row
        
        return curr_row[-1]


def _neh_standard(P):
    """Standard NEH heuristic."""
    n_jobs = P.shape[0]
    totals = P.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [int(j)] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_first_last(P):
    """NEH variant using first machine processing time for sorting."""
    n_jobs = P.shape[0]
    sort_key = P[:, 0]
    sorted_jobs = np.argsort(-sort_key)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [int(j)] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_last_machine(P):
    """NEH variant using last machine processing time for sorting."""
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    sort_key = P[:, n_machines - 1]
    sorted_jobs = np.argsort(-sort_key)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [int(j)] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_random(P, rng):
    """NEH variant with random sorting."""
    n_jobs = P.shape[0]
    sorted_jobs = rng.permutation(n_jobs)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [int(j)] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _insertion_search(order, P, max_passes=20, time_limit=None, start_time=None):
    """Insertion local search: try moving each job to every other position."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        for i in range(n):
            if time_limit is not None and start_time is not None:
                if time.time() > start_time + time_limit:
                    return best_order, best_ms
            job = best_order[i]
            # Remove job from position i
            remaining = best_order[:i] + best_order[i+1:]
            best_pos = i
            best_candidate_ms = best_ms
            
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(candidate, P)
                if ms < best_candidate_ms - 1e-10:
                    best_candidate_ms = ms
                    best_pos = pos
            
            if best_pos != i:
                remaining = best_order[:i] + best_order[i+1:]
                best_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_ms = best_candidate_ms
                improved = True
        
        if not improved:
            break
    
    return best_order, best_ms


def _or_opt_search(order, P, max_passes=10, time_limit=None, start_time=None):
    """Or-opt local search: try moving blocks of size 1-3 to other positions."""
    n = len(order)
    if n <= 3:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        # Try blocks of size 1, 2, 3
        for block_size in [1, 2, 3]:
            if n < block_size + 1:
                continue
            for i in range(n - block_size + 1):
                if time_limit is not None and start_time is not None:
                    if time.time() > start_time + time_limit:
                        return best_order, best_ms
                block = best_order[i:i+block_size]
                # Remove block
                remaining = best_order[:i] + best_order[i+block_size:]
                # Try inserting at all other positions
                for pos in range(len(remaining) + 1):
                    # Skip positions that would result in same or adjacent placement
                    if pos == i or (block_size == 1 and pos == i + 1) or \
                       (block_size > 1 and pos >= i and pos < i + block_size):
                        continue
                    candidate = remaining[:pos] + block + remaining[pos:]
                    ms = _makespan(candidate, P)
                    if ms < best_ms - 1e-10:
                        best_order = candidate
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order, best_ms


def _order_crossover(p1, p2, rng):
    """Order Crossover (OX) for permutations."""
    n = len(p1)
    if n < 2:
        return list(p1)
    
    # Choose two cut points
    i = rng.randint(0, n - 1)
    j = rng.randint(0, n - 1)
    if i > j:
        i, j = j, i
    
    # Segment from p1
    segment = p1[i:j+1]
    segment_set = set(segment)
    
    # Fill remaining positions with elements from p2 in order
    remaining = [x for x in p2 if x not in segment_set]
    
    child = [None] * n
    child[i:j+1] = segment
    
    # Fill positions before i (from end of remaining) and after j (from start)
    # Actually, standard OX: fill from j+1 onwards, then wrap around to i-1
    pos = j + 1
    idx = 0
    while pos != i and idx < len(remaining):
        child[pos] = remaining[idx]
        pos = (pos + 1) % n
        idx += 1
    
    # Fill remaining positions
    while idx < len(remaining):
        if child[pos] is None:
            child[pos] = remaining[idx]
            idx += 1
        pos = (pos + 1) % n
    
    return child


def _mutation(order, rng, mutation_rate=0.1):
    """Apply swap and insertion mutations."""
    n = len(order)
    if n < 2:
        return list(order)
    
    child = list(order)
    
    # Swap mutation
    if rng.random() < mutation_rate:
        i, j = rng.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    
    # Insertion mutation
    if rng.random() < mutation_rate:
        i = rng.randint(0, n - 1)
        job = child[i]
        child.pop(i)
        j = rng.randint(0, len(child))
        child.insert(j, job)
    
    # Or-opt mutation
    if rng.random() < mutation_rate and n >= 3:
        block_len = rng.choice([1, 2, 3])
        if n >= block_len + 1:
            i = rng.randint(0, n - block_len)
            block = child[i:i+block_len]
            child = child[:i] + child[i+block_len:]
            j = rng.randint(0, len(child))
            child = child[:j] + block + child[j:]
    
    return child


def _genetic_algorithm(population, P, start_time, time_limit, rng):
    """Genetic Algorithm with Order Crossover and mutation."""
    n_jobs = len(population[0])
    pop_size = len(population)
    
    # Evaluate initial population
    evaluated = []
    for ind in population:
        ms = _makespan(ind, P)
        evaluated.append((ms, ind))
    evaluated.sort(key=lambda x: x[0])
    
    best_order = evaluated[0][1][:]
    best_ms = evaluated[0][0]
    
    # GA parameters
    elite_size = max(2, pop_size // 5)
    tournament_size = 3
    mutation_rate = 0.15
    max_generations = 200
    
    generation = 0
    while generation < max_generations:
        if time.time() > start_time + time_limit:
            break
        
        generation += 1
        
        # Create new population
        new_pop = []
        
        # Elitism: keep best individuals
        for i in range(min(elite_size, len(evaluated))):
            new_pop.append(evaluated[i][1][:])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() > start_time + time_limit:
                break
            
            # Tournament selection
            def tournament_select():
                candidates = rng.sample(range(len(evaluated)), min(tournament_size, len(evaluated)))
                best_idx = min(candidates, key=lambda x: evaluated[x][0])
                return evaluated[best_idx][1]
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover
            if rng.random() < 0.9:  # 90% crossover rate
                child = _order_crossover(p1, p2, rng)
            else:
                child = p1[:]
            
            # Mutation
            child = _mutation(child, rng, mutation_rate)
            
            new_pop.append(child)
        
        # Evaluate new population
        evaluated = []
        for ind in new_pop:
            ms = _makespan(ind, P)
            evaluated.append((ms, ind))
        
        evaluated.sort(key=lambda x: x[0])
        
        # Update best
        if evaluated[0][0] < best_ms:
            best_ms = evaluated[0][0]
            best_order = evaluated[0][1][:]
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = P.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)
    
    # Generate multiple NEH variants to seed the population
    candidates = []
    candidates.append(_neh_standard(P))
    candidates.append(_neh_first_last(P))
    candidates.append(_neh_last_machine(P))
    
    # Add more random NEH variants for diversity
    for _ in range(5):
        candidates.append(_neh_random(P, rng))
    
    # Evaluate all candidates
    best_order = None
    best_ms = float('inf')
    for cand in candidates:
        ms = _makespan(cand, P)
        if ms < best_ms:
            best_ms = ms
            best_order = cand
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_order
    
    # Phase 1: Quick insertion search on best candidate to get a good starting point
    remaining_time = deadline - (time.time() - start_time) - 5.0
    if remaining_time > 2:
        best_order, best_ms = _insertion_search(
            best_order, P, max_passes=5, time_limit=remaining_time, start_time=start_time
        )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        return best_order
    
    # Phase 2: Genetic Algorithm
    # Build population from diverse NEH variants
    population = []
    for cand in candidates:
        population.append(list(cand))
    
    # Ensure population has at least 10 individuals
    while len(population) < 10:
        # Add mutated versions of best
        mutated = _mutation(best_order, py_rng, mutation_rate=0.3)
        population.append(mutated)
    
    # Limit population size for larger problems
    max_pop = min(len(population), max(10, n_jobs // 2))
    population = population[:max_pop]
    
    remaining_time = deadline - (time.time() - start_time) - 5.0
    if remaining_time > 5:
        ga_order, ga_ms = _genetic_algorithm(
            population, P, start_time, remaining_time, py_rng
        )
        if ga_ms < best_ms:
            best_order = ga_order
            best_ms = ga_ms
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_order
    
    # Phase 3: Or-opt search on best GA result
    remaining_time = deadline - (time.time() - start_time) - 3.0
    if remaining_time > 2:
        best_order, best_ms = _or_opt_search(
            best_order, P, max_passes=8, time_limit=remaining_time, start_time=start_time
        )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 4: Final insertion polish
    remaining_time = deadline - (time.time() - start_time) - 1.0
    if remaining_time > 2:
        best_order, best_ms = _insertion_search(
            best_order, P, max_passes=15, time_limit=remaining_time, start_time=start_time
        )
    
    return best_order
# EVOLVE-BLOCK-END
