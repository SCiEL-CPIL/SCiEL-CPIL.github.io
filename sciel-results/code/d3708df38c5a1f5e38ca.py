"""Flow-shop permutation via multi-start heuristics + VND local search + Iterated Local Search."""
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
            # prev[m] is completion time of previous job on machine m
            # curr[m-1] is start time of current job on machine m (which is completion on m-1)
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

def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm

def _local_search_vnd(pt_list, perm, deadline, max_rounds=5000):
    """
    Variable Neighborhood Descent (VND) using Insertion and 2-Opt (Reversal).
    Alternates between neighborhoods or switches when stuck.
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    # Tabu list for recent moves to avoid cycling (optional but helpful)
    # For simplicity and speed, we rely on first-improvement in VND loops.
    
    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Neighborhood 1: Insertion
        # Try to find any improving insertion
        for i in range(n):
            if time.time() > deadline:
                break
            job = current[i]
            rest = current[:i] + current[i+1:]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
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
            # Neighborhood 2: 2-Opt Reversal
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

def _perturb(pt_list, perm, intensity, rng):
    """
    Perturb the permutation by performing 'intensity' random insertion moves.
    """
    n = len(perm)
    current = list(perm)
    
    # Adaptive intensity: usually between 1 and 5% of n
    num_moves = max(1, int(n * intensity))
    
    for _ in range(num_moves):
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        current = rest[:j] + [job] + rest[j:]
        
    return current

def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    # Safety margin: 150s limit, stop at 145s
    deadline = start + 145.0 

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()
    
    # 1. Generate diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
    ]
    # Add a few random starts
    for s in range(3):
        candidates.append(_random_initial(n_jobs, 42 + s))

    # 2. Evaluate and select best initial
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # 3. Initial Local Search on the best candidate
    # Use a shorter deadline for the first polish to ensure we enter ILS loop
    initial_deadline = min(deadline - 10.0, start + 5.0) # Max 5s for first LS
    if time.time() < initial_deadline:
        best_perm, best_ms = _local_search_vnd(pt_list, best_perm, initial_deadline)

    # 4. Iterated Local Search (ILS)
    rng = random.Random(123)
    
    # Track recent best to avoid wasting time on bad perturbations
    # We will run ILS until time runs out.
    
    while time.time() < deadline - 2.0:
        # Perturb the current best solution
        # Intensity can be adapted, e.g., 0.05 * n_jobs
        intensity = 0.05 
        perturbed = _perturb(pt_list, best_perm, intensity, rng)
        
        # Local Search on the perturbed solution
        # Reserve some time for the next iteration
        iter_deadline = deadline - 2.0
        # If we have very little time, skip LS and just keep best
        if time.time() > iter_deadline - 1.0:
             break
            
        new_perm, new_ms = _local_search_vnd(pt_list, perturbed, iter_deadline)
        
        # Accept if better, or with some probability if worse (to escape)
        # For Flow Shop, usually "Accept if better or equal" is safe, 
        # but ILS often accepts worse solutions to explore. 
        # Let's use a simple rule: Accept if better. If not, keep old best.
        # To be more aggressive, we can accept worse solutions if they are not too bad, 
        # but without a SA-like temperature, it's hard. 
        # Standard ILS accepts the new local optimum if it's better than the current global best.
        # Or it accepts the new local optimum as the new "current" for the next perturbation 
        # even if it's worse than global best, to explore new basins.
        
        # Strategy: Always set 'current' to the new local optimum for the next perturbation.
        # Update 'global_best' only if improved.
        # This allows escaping local optima.
        
        # However, if the new solution is much worse, we might wander.
        # Let's stick to: Update global best if improved. 
        # For the next iteration, we perturb the GLOBAL BEST again. 
        # This is "Restart ILS". It's simpler and often effective.
        # Alternative: Perturb the NEW local optimum. This is "Standard ILS".
        
        # Let's use Standard ILS: Perturb the result of the last LS.
        # If the result is much worse, we might want to revert to global best.
        # Let's accept the new solution as the base for the next iteration ONLY if it's not drastically worse.
        # Or simply: Always use the new local optimum as the base for the next perturbation.
        
        if new_ms < best_ms:
            best_ms = new_ms
            best_perm = new_perm
            
        # For the next iteration, we will perturb 'best_perm' (the global best) 
        # to ensure we are always exploring around the best known solution.
        # This is safer than wandering into bad basins.
        
    # 5. Final Polish
    if time.time() < deadline - 0.5:
        final_perm, final_ms = _local_search_vnd(pt_list, best_perm, deadline)
        if final_ms < best_ms:
            best_ms = final_ms
            best_perm = final_perm

    return best_perm
# EVOLVE-BLOCK-END
