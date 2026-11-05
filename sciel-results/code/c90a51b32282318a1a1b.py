"""Flow-shop permutation via multi-start heuristics + fast insertion local search + tabu search with block moves."""
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


def _makespan_with_early_termination(pt_list, perm, upper_bound):
    """Makespan with early termination if any machine exceeds upper_bound."""
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
        if curr[0] > upper_bound:
            return curr[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
            if curr[m] > upper_bound:
                return curr[m]
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


def _local_search_insertion(pt_list, perm, deadline, max_rounds=5000):
    """
    Local search using insertion neighborhood: remove job at position i, insert at position j.
    Uses early termination for makespan evaluation.
    """
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
                ms = _makespan_with_early_termination(pt_list, test, current_ms)
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


def _tabu_search(pt_list, perm, deadline, max_iter=200000, tabu_tenure=7, block_size=3, seed=42):
    """
    Tabu Search with block-adjacent insertion neighborhood.
    Moves: take a block of 1-3 consecutive jobs and insert at any other position.
    Uses dynamic tabu list to prevent cycling.
    """
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    # Tabu list: store (start_pos, end_pos, insert_pos) tuples
    tabu_list = []
    tabu_counter = {}
    
    # Diversification: if stuck for too long, add random perturbation
    stagnation_count = 0
    max_stagnation = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a random move from the block-adjacent neighborhood
        # Choose block size (1 to block_size)
        max_block = min(block_size, n - 1)
        block_len = rng.randint(1, max_block)
        
        # Choose start position of block
        start = rng.randint(0, n - block_len)
        end = start + block_len - 1
        
        # Choose insertion position (excluding original position)
        valid_positions = list(range(n - block_len + 1))
        valid_positions.remove(start)
        if not valid_positions:
            continue
        insert_pos = rng.choice(valid_positions)
        
        # Create the new permutation
        block = current[start:end+1]
        rest = current[:start] + current[end+1:]
        new_perm = rest[:insert_pos] + block + rest[insert_pos:]
        
        # Check if this move is tabu
        move_key = (start, end, insert_pos)
        is_tabu = move_key in tabu_counter
        
        # Evaluate new makespan
        new_ms = _makespan_with_early_termination(pt_list, new_perm, current_ms if not is_tabu else float('inf'))
        
        # Acceptance criteria:
        # 1. If not tabu and improves, accept
        # 2. If tabu but improves best (aspiration criterion), accept
        # 3. If not tabu and doesn't improve, accept with some probability (diversification)
        
        should_accept = False
        if not is_tabu:
            if new_ms < current_ms - 1e-9:
                should_accept = True
            elif new_ms < best_ms - 1e-9:
                should_accept = True
            else:
                # Small probability of accepting non-improving moves for diversification
                if rng.random() < 0.05:
                    should_accept = True
        else:
            # Aspiration: accept tabu move if it improves the best
            if new_ms < best_ms - 1e-9:
                should_accept = True
        
        if should_accept:
            current = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best = list(current)
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Add to tabu list
            tabu_counter[move_key] = tabu_tenure
            tabu_list.append(move_key)
            
            # Decay tabu counters
            for key in list(tabu_counter.keys()):
                tabu_counter[key] -= 1
                if tabu_counter[key] <= 0:
                    del tabu_counter[key]
            
            # If stuck for too long, reset and add random perturbation
            if stagnation_count > max_stagnation:
                # Random restart with perturbation
                for _ in range(3):
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    if i != j:
                        current[i], current[j] = current[j], current[i]
                current_ms = _makespan_fast(pt_list, current)
                if current_ms < best_ms:
                    best_ms = current_ms
                    best = list(current)
                stagnation_count = 0
                # Clear tabu list
                tabu_counter.clear()
                tabu_list.clear()
                
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
        new_ms = _makespan_with_early_termination(pt_list, test, current_ms)
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

    # Multi-start local search: polish top candidates
    for ms0, cand in candidate_results:
        if time.time() > deadline - 30.0:
            break
        # Skip if too far from best
        if ms0 > best_ms * 1.2:
            continue

        # Insertion local search
        perm, ms = _local_search_insertion(pt_list, cand, deadline - 30.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Tabu search from best solution
    if time.time() < deadline - 15.0:
        perm, ms = _tabu_search(pt_list, best_perm, deadline - 15.0, max_iter=300000, tabu_tenure=8, block_size=3, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 8.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 8.0, max_iter=100000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish with insertion local search
    if time.time() < deadline - 3.0:
        perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 1.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final tabu polish with smaller block size for fine-tuning
    if time.time() < deadline - 0.5:
        perm, ms = _tabu_search(pt_list, best_perm, deadline - 0.5, max_iter=50000, tabu_tenure=5, block_size=2, seed=13)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
