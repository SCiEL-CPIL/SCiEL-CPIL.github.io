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

def _tsp_nearest_neighbor(pt):
    """TSP-like construction: start with highest total job, then greedily add the job with minimum incremental makespan."""
    n, m = pt.shape
    if n <= 1:
        return np.arange(n)
    
    total_pt = pt.sum(axis=1)
    start_job = int(np.argmax(total_pt))
    
    remaining = set(range(n))
    remaining.remove(start_job)
    perm = [start_job]
    
    while remaining:
        best_job = -1
        best_ms = float('inf')
        for job in remaining:
            # Try adding job at the end
            trial = perm + [job]
            ms = _makespan(trial, pt)
            if ms < best_ms:
                best_ms = ms
                best_job = job
        perm.append(best_job)
        remaining.remove(best_job)
    
    return np.array(perm, dtype=int)

def _insertion_ls(perm, pt, time_limit, start_time):
    """Insertion local search with fast makespan evaluation."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    comp_cache = np.zeros(pt.shape[1], dtype=float)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            remaining = np.delete(best_perm, i)
            for j in range(n):
                if time.time() - start_time >= time_limit:
                    break
                new_perm = np.insert(remaining, j, job)
                ms = _makespan_fast(new_perm, pt, comp_cache)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    i = j
                    break
    
    return best_perm, best_ms

def _or_opt_ls(perm, pt, time_limit, start_time):
    """Or-opt: move a segment of 1-3 jobs to a better position."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    comp_cache = np.zeros(pt.shape[1], dtype=float)
    
    improved = True
    seg_lens = [1, 2, 3]
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        for seg_len in seg_lens:
            if time.time() - start_time >= time_limit:
                break
            for i in range(n - seg_len + 1):
                if time.time() - start_time >= time_limit:
                    break
                segment = best_perm[i:i+seg_len]
                remaining = np.concatenate([best_perm[:i], best_perm[i+seg_len:]])
                for j in range(len(remaining) + 1):
                    if time.time() - start_time >= time_limit:
                        break
                    new_perm = np.insert(remaining, j, segment)
                    ms = _makespan_fast(new_perm, pt, comp_cache)
                    if ms < best_ms:
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        # Restart search from beginning
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_perm, best_ms

def _reversal_ls(perm, pt, time_limit, start_time):
    """Segment reversal local search."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    comp_cache = np.zeros(pt.shape[1], dtype=float)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 2, n):
                if time.time() - start_time >= time_limit:
                    break
                new_perm = best_perm.copy()
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                ms = _makespan_fast(new_perm, pt, comp_cache)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    break
            if improved:
                break
    
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
    
    # TSP nearest neighbor
    perm_tsp = _tsp_nearest_neighbor(processing_times)
    initial_solutions.append(perm_tsp)
    
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
    for _ in range(8):
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
    
    # Local search phases
    # Phase 1: Insertion LS
    best_perm, best_ms = _insertion_ls(best_perm, processing_times, 20.0, start_time)
    
    # Phase 2: Or-opt LS
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _or_opt_ls(best_perm, processing_times, 20.0, start_time)
    
    # Phase 3: Reversal LS
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _reversal_ls(best_perm, processing_times, 15.0, start_time)
    
    # Phase 4: Second insertion pass
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _insertion_ls(best_perm, processing_times, 15.0, start_time)
    
    # Phase 5: Final or-opt polish
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _or_opt_ls(best_perm, processing_times, 10.0, start_time)
    
    # Random restarts if time permits
    remaining_time = effective_deadline - (time.time() - start_time)
    if remaining_time > 15.0:
        num_restarts = min(4, int(remaining_time // 25))
        for _ in range(num_restarts):
            if time.time() - start_time >= effective_deadline:
                break
            perm_rand = rng.permutation(n)
            perm_rand, ms_rand = _insertion_ls(perm_rand, processing_times, 15.0, start_time)
            if ms_rand < best_ms:
                best_ms = ms_rand
                best_perm = perm_rand.copy()
            if time.time() - start_time >= effective_deadline:
                break
            perm_rand, ms_rand = _or_opt_ls(perm_rand, processing_times, 10.0, start_time)
            if ms_rand < best_ms:
                best_ms = ms_rand
                best_perm = perm_rand.copy()
    
    return list(best_perm)
# EVOLVE-BLOCK-END
