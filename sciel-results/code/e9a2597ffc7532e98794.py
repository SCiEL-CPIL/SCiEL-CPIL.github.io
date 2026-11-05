"""Flow-shop permutation via CDS+NEH multi-start, fast insertion local search, and segment reversal."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """Fast makespan calculation using pure Python lists."""
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


def _cds_neh_initial(pt):
    """CDS (Campbell, Dudek, Smith) heuristic: take best of NEH on full, first half, and second half jobs."""
    n_jobs, n_machines = pt.shape
    pt_list = pt.tolist()
    
    # Full NEH
    full_perm = _neh_initial(pt)
    full_ms = _makespan_fast(pt_list, full_perm)
    
    best_perm = full_perm
    best_ms = full_ms
    
    # First half
    if n_jobs >= 4:
        half = n_jobs // 2
        totals = pt[:half].sum(axis=1)
        order = list(np.argsort(-totals))
        perm = [order[0]]
        for i in range(1, half):
            job = order[i]
            best_pos = 0
            best_ms_i = float('inf')
            for pos in range(len(perm) + 1):
                test = perm[:pos] + [job] + perm[pos:]
                ms = _makespan_fast(pt_list, test)
                if ms < best_ms_i:
                    best_ms_i = ms
                    best_pos = pos
            perm.insert(best_pos, job)
        
        # Now insert remaining jobs using NEH
        remaining = list(range(half, n_jobs))
        remaining.sort(key=lambda j: -pt[j].sum())
        for job in remaining:
            best_pos = 0
            best_ms_i = float('inf')
            for pos in range(len(perm) + 1):
                test = perm[:pos] + [job] + perm[pos:]
                ms = _makespan_fast(pt_list, test)
                if ms < best_ms_i:
                    best_ms_i = ms
                    best_pos = pos
            perm.insert(best_pos, job)
        
        ms = _makespan_fast(pt_list, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Second half
    if n_jobs >= 4:
        half = n_jobs // 2
        totals = pt[half:].sum(axis=1)
        order = list(np.argsort(-totals))
        perm = [half + order[0]]
        for i in range(1, half):
            job = half + order[i]
            best_pos = 0
            best_ms_i = float('inf')
            for pos in range(len(perm) + 1):
                test = perm[:pos] + [job] + perm[pos:]
                ms = _makespan_fast(pt_list, test)
                if ms < best_ms_i:
                    best_ms_i = ms
                    best_pos = pos
            perm.insert(best_pos, job)
        
        remaining = list(range(0, half))
        remaining.sort(key=lambda j: -pt[j].sum())
        for job in remaining:
            best_pos = 0
            best_ms_i = float('inf')
            for pos in range(len(perm) + 1):
                test = perm[:pos] + [job] + perm[pos:]
                ms = _makespan_fast(pt_list, test)
                if ms < best_ms_i:
                    best_ms_i = ms
                    best_pos = pos
            perm.insert(best_pos, job)
        
        ms = _makespan_fast(pt_list, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    return best_perm


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


def _local_search_insertion(pt_list, perm, deadline, max_rounds=2000):
    """Local search using insertion neighborhood with first-improvement."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = current[i]
            rest = current[:i] + current[i+1:]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                if time.time() > deadline:
                    break
                test = rest[:j] + [job] + rest[j:]
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


def _local_search_2opt(pt_list, perm, deadline, max_rounds=2000):
    """Local search using 2-opt segment reversal."""
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


def _simulated_annealing(pt_list, perm, deadline, max_iter=50000, seed=0):
    """Simulated annealing with insertion moves."""
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
    cooling = 0.999

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
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


def _ox_crossover(pt_list, p1, p2, rng):
    """Order Crossover (OX) for permutations."""
    n = len(p1)
    if n <= 2:
        return list(p1)
    
    i, j = sorted(rng.sample(range(n), 2))
    
    child = [None] * n
    child[i:j+1] = p1[i:j+1]
    
    segment_set = set(p1[i:j+1])
    fill_pos = 0
    for job in p2:
        if job not in segment_set:
            while child[fill_pos] is not None:
                fill_pos += 1
            child[fill_pos] = job
            fill_pos += 1
            if fill_pos >= n:
                fill_pos = 0
    
    return child


def _ga_with_ox(pt_list, initial_perms, deadline, pop_size=20, max_gen=50, seed=42):
    """Genetic Algorithm with Order Crossover."""
    n = len(initial_perms[0]) if initial_perms else 0
    if n <= 1:
        return initial_perms[0], _makespan_fast(pt_list, initial_perms[0]) if initial_perms else ([], 0.0)
    
    rng = random.Random(seed)
    
    population = list(initial_perms)
    while len(population) < pop_size:
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(perm)
    
    scores = [_makespan_fast(pt_list, p) for p in population]
    
    best_perm = population[scores.index(min(scores))]
    best_ms = min(scores)
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        new_population = []
        
        sorted_idx = np.argsort(scores)
        for idx in sorted_idx[:2]:
            new_population.append(population[idx])
        
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            def tournament_select():
                idx1 = rng.randint(0, len(population) - 1)
                idx2 = rng.randint(0, len(population) - 1)
                return population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            if rng.random() < 0.8:
                child = _ox_crossover(pt_list, p1, p2, rng)
            else:
                child = list(p1)
            
            if rng.random() < 0.3:
                i = rng.randint(0, n - 1)
                job = child[i]
                rest = child[:i] + child[i+1:]
                j = rng.randint(0, n - 1)
                if j == i or j == i + 1:
                    j = (j + 1) % n
                child = rest[:j] + [job] + rest[j:]
            
            new_population.append(child)
        
        population = new_population
        scores = [_makespan_fast(pt_list, p) for p in population]
        
        gen_best_idx = scores.index(min(scores))
        if scores[gen_best_idx] < best_ms:
            best_ms = scores[gen_best_idx]
            best_perm = population[gen_best_idx]
        
        if time.time() < deadline - 5.0:
            polished, polished_ms = _local_search_insertion(pt_list, best_perm, deadline - 5.0, max_rounds=200)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished
    
    return best_perm, best_ms


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

    # Build diverse initial solutions including CDS
    candidates = [
        _cds_neh_initial(pt),
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
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

    # Sort candidates by makespan
    candidate_results = []
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        candidate_results.append((ms, cand))
    candidate_results.sort(key=lambda x: x[0])

    # Multi-start local search
    for ms0, cand in candidate_results:
        if time.time() > deadline - 15.0:
            break
        if ms0 > best_ms * 1.2:
            continue

        perm, ms = _local_search_insertion(pt_list, cand, deadline - 15.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

        if time.time() < deadline - 10.0:
            perm2, ms2 = _local_search_2opt(pt_list, perm, deadline - 10.0)
            if ms2 < best_ms:
                best_ms = ms2
                best_perm = perm2

    # GA with OX crossover
    if time.time() < deadline - 15.0:
        top_candidates = [cand for _, cand in candidate_results[:10]]
        if len(top_candidates) < 5:
            top_candidates.extend(candidates)
        
        perm, ms = _ga_with_ox(pt_list, top_candidates, deadline - 15.0, pop_size=20, max_gen=30, seed=42)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Simulated annealing
    if time.time() < deadline - 10.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 10.0, max_iter=50000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish
    if time.time() < deadline - 3.0:
        perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 2.0, max_rounds=500)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    if time.time() < deadline - 1.0:
        perm, ms = _local_search_2opt(pt_list, best_perm, deadline - 0.5, max_rounds=500)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
