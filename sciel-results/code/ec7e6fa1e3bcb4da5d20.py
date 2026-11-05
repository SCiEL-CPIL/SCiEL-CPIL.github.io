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
    This is used to guide faster neighborhood evaluation.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    costs = [[0.0] * n_jobs for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                continue
            # Marginal cost: sum of max(0, pt[i][m] - pt[j][m]) for m>=1 plus pt[i][0]
            # This is a heuristic: the extra load job i adds when following job j
            row_i = pt_list[i]
            row_j = pt_list[j]
            c = row_i[0]
            for m in range(1, n_machines):
                diff = row_i[m] - row_j[m]
                if diff > 0:
                    c += diff
            costs[i][j] = c
    return costs


def _makespan_vectorized(pt, perm):
    """Vectorized makespan using numpy."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # pt is shape [n_jobs, n_machines]
    # perm is list of job indices
    # We need to compute the completion times
    
    # C[j][m] for job j in position j of perm
    # C[0][0] = pt[perm[0]][0]
    # C[j][0] = C[j-1][0] + pt[perm[j]][0]
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + pt[perm[j]][m]
    
    # Let's use a loop over jobs but vectorize over machines
    # This is O(n_jobs * n_machines) but with numpy overhead per job
    # For small n_machines, the Python loop in _makespan_fast is faster.
    # For large n_machines, this might be better?
    # Actually, _makespan_fast is quite optimized in Python.
    # Let's stick to _makespan_fast for consistency and speed in local search.
    pass


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


def _local_search_insertion(pt_list, perm, deadline, max_rounds=5000, cost_matrix=None):
    """
    Local search using insertion neighborhood: remove job at position i, insert at position j.
    Uses Top-K improvement strategy for better local optima.
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    # Dynamic K based on problem size
    if n < 10:
        K = 10
    elif n < 20:
        K = 5
    else:
        K = 3

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # Build list of (estimated_cost, i, j) for all valid moves
        moves = []
        for i in range(n):
            job = current[i]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                if cost_matrix is not None:
                    # Estimate: cost of job after its new predecessor
                    pred = current[j-1] if j > 0 else current[0]
                    est = cost_matrix[job][pred]
                else:
                    est = 0.0
                moves.append((est, i, j))
        
        if not moves:
            break

        moves.sort(key=lambda x: x[0])
        
        # Evaluate top K moves
        best_move_ms = current_ms
        best_move_i = -1
        best_move_j = -1
        
        for idx in range(min(K, len(moves))):
            est, i, j = moves[idx]
            if time.time() > deadline:
                break
            job = current[i]
            rest = current[:i] + current[i+1:]
            test = rest[:j] + [job] + rest[j:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_move_ms - 1e-9:
                best_move_ms = ms
                best_move_i = i
                best_move_j = j
        
        if best_move_i != -1:
            # Apply the best move found
            job = current[best_move_i]
            rest = current[:best_move_i] + current[best_move_i+1:]
            current = rest[:best_move_j] + [job] + rest[best_move_j:]
            current_ms = best_move_ms
            improved = True
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best = list(current)
        
        if not improved:
            break
            
    return best, best_ms


def _local_search_2opt(pt_list, perm, deadline, max_rounds=5000, cost_matrix=None):
    """
    Local search using 2-opt segment reversal: reverse the segment from i to j.
    If cost_matrix is provided, skip reversals that are unlikely to help.
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
                # Quick heuristic skip: if the segment is already in "good" order
                # based on cost matrix, skip this reversal
                if cost_matrix is not None and (j - i) > 2:
                    # Check if reversing would likely worsen things
                    # Heuristic: sum of costs in current order vs reversed
                    # For speed, just check endpoints
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
                # Reverse segment [i..j]
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
        # Bias j selection: prefer positions where the cost is lower
        if cost_matrix is not None:
            # Sample j with probability proportional to exp(-cost/T)
            j = rng.randint(0, n - 1)
            # With some probability, try a better position
            if rng.random() < 0.3:
                # Find the position with lowest estimated cost
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

    # Multi-start local search: polish top candidates
    for ms0, cand in candidate_results:
        if time.time() > deadline - 15.0:
            break
        # Skip if too far from best
        if ms0 > best_ms * 1.2:
            continue

        # Insertion local search
        perm, ms = _local_search_insertion(pt_list, cand, deadline - 15.0, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

        # 2-opt segment reversal if time permits
        if time.time() < deadline - 10.0:
            perm2, ms2 = _local_search_2opt(pt_list, perm, deadline - 10.0, cost_matrix=cost_matrix)
            if ms2 < best_ms:
                best_ms = ms2
                best_perm = perm2

    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 8.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 8.0, max_iter=200000, seed=7, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final polish with insertion local search
    if time.time() < deadline - 2.0:
        perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 1.0, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final 2-opt polish
    if time.time() < deadline - 0.5:
        perm, ms = _local_search_2opt(pt_list, best_perm, deadline - 0.5, cost_matrix=cost_matrix)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
