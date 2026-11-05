# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import permutations


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    n = len(perm)
    m = pt.shape[1]
    # Completion time on each machine for each job in sequence
    # C[j][k] = completion time of job j on machine k
    # C[j][0] = sum_{i=0}^{j} pt[perm[i]][0]
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + pt[perm[j]][k]
    
    # Efficient computation
    c_prev = np.zeros(m)  # completion times of previous job on each machine
    makespan = 0.0
    for j in range(n):
        job = perm[j]
        c_curr = np.empty(m)
        for k in range(m):
            if k == 0:
                c_curr[k] = c_curr[k-1] + pt[job, k] if k > 0 else pt[job, k]
            else:
                c_curr[k] = max(c_prev[k], c_curr[k-1]) + pt[job, k]
        c_prev = c_curr
        makespan = c_prev[-1]
    return makespan


def _makespan_fast(perm, pt):
    """Vectorized makespan computation."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # Build the matrix of processing times in the order of the permutation
    p = pt[perm]  # shape (n, m)
    
    # Cumulative sums along jobs for machine 0
    c0 = np.cumsum(p[:, 0])
    
    if m == 1:
        return float(c0[-1])
    
    # For machines 1..m-1, we need:
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + p[j][k]
    # This is a dynamic programming recurrence.
    
    # We'll compute column by column
    c = np.zeros((n, m))
    c[:, 0] = c0
    
    for k in range(1, m):
        c[0, k] = c[0, k-1] + p[0, k]
        for j in range(1, n):
            c[j, k] = max(c[j, k-1], c[j-1, k]) + p[j, k]
    
    return float(c[-1, -1])


def _neh_order(pt):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop."""
    n = pt.shape[0]
    # Sort by total processing time in descending order
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insert each job at the best position
    result = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(result) + 1):
            candidate = result[:pos] + [job] + result[pos:]
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        result.insert(best_pos, job)
    
    return result


def _two_opt(perm, pt):
    """Two-opt local search: try all pairwise swaps, keep best improvement."""
    n = len(perm)
    if n < 2:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if i == j:
                    continue
                candidate = list(best_perm)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan_fast(candidate, pt)
                if ms < best_ms - 1e-12:
                    best_perm = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return best_perm


def _adjacent_swap(perm, pt):
    """Adjacent swap local search."""
    n = len(perm)
    if n < 2:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            candidate = list(best_perm)
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms - 1e-12:
                best_perm = candidate
                best_ms = ms
                improved = True
                break
    
    return best_perm


def _vnd(perm, pt, deadline):
    """Variable Neighborhood Descent: adjacent swaps -> 2-opt -> block insertions."""
    n = len(perm)
    if n < 2:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Neighborhood 1: adjacent swaps
    improved = True
    while improved:
        if time.time() > deadline:
            return best_perm
        improved = False
        for i in range(n - 1):
            candidate = list(best_perm)
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms - 1e-12:
                best_perm = candidate
                best_ms = ms
                improved = True
                break
    
    # Neighborhood 2: 2-opt (all pairwise swaps)
    if time.time() > deadline:
        return best_perm
    improved = True
    while improved:
        if time.time() > deadline:
            return best_perm
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                candidate = list(best_perm)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan_fast(candidate, pt)
                if ms < best_ms - 1e-12:
                    best_perm = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    # Neighborhood 3: block insertions (move a block of 1-3 jobs to another position)
    if time.time() > deadline:
        return best_perm
    for block_size in [1, 2, 3]:
        if time.time() > deadline:
            return best_perm
        improved = True
        while improved:
            if time.time() > deadline:
                return best_perm
            improved = False
            for i in range(n - block_size + 1):
                block = best_perm[i:i+block_size]
                remaining = best_perm[:i] + best_perm[i+block_size:]
                for pos in range(len(remaining) + 1):
                    if pos == i:  # skip same position
                        continue
                    candidate = remaining[:pos] + block + remaining[pos:]
                    ms = _makespan_fast(candidate, pt)
                    if ms < best_ms - 1e-12:
                        best_perm = candidate
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
    
    return best_perm


def _random_permutation(n):
    """Generate a random permutation of [0, n-1]."""
    perm = list(range(n))
    random.shuffle(perm)
    return perm


def _gsa_step(perm, pt, n_particles, deadline):
    """
    Gravitational Search Algorithm step.
    Particles are permutations; fitness is makespan (lower is better).
    We use a simplified GSA where we generate perturbations guided by
    the best and worst solutions.
    """
    n = len(perm)
    m = pt.shape[1]
    
    # Initialize particles
    particles = [list(perm)]
    for _ in range(n_particles - 1):
        # Create perturbed version
        p = list(perm)
        # Random swaps
        for _ in range(random.randint(1, min(3, n - 1))):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        particles.append(p)
    
    # Evaluate all particles
    fitnesses = [_makespan_fast(p, pt) for p in particles]
    
    best_idx = int(np.argmin(fitnesses))
    worst_idx = int(np.argmax(fitnesses))
    best_particle = list(particles[best_idx])
    best_fitness = fitnesses[best_idx]
    
    # Generate new candidates guided by gravity-like attraction to best
    new_candidates = []
    for _ in range(n_particles):
        # Mix between best particle and a random particle
        p = list(particles[random.randint(0, n_particles - 1)])
        
        # Attraction: swap positions to move toward best
        for i in range(n):
            if p[i] != best_particle[i]:
                # Try to place best_particle[i] at position i
                if random.random() < 0.5:
                    # Find where best_particle[i] is in p
                    for j in range(n):
                        if p[j] == best_particle[i]:
                            p[i], p[j] = p[j], p[i]
                            break
    
        new_candidates.append(p)
    
    # Evaluate new candidates
    for c in new_candidates:
        if time.time() > deadline:
            break
        ms = _makespan_fast(c, pt)
        if ms < best_fitness:
            best_particle = list(c)
            best_fitness = ms
    
    return best_particle, best_fitness


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Uses a combination of:
    - NEH heuristic for initial solution
    - VND (Variable Neighborhood Descent) for local optimization
    - Gravitational Search Algorithm for exploration
    - Multiple restarts with time budget management
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times)
    n, m = pt.shape
    
    if n <= 1:
        return list(range(n))
    
    if n <= 8:
        # Small enough for exhaustive or near-exhaustive search
        best_perm = None
        best_ms = float('inf')
        # Use NEH as starting point
        neh = _neh_order(pt)
        best_perm = list(neh)
        best_ms = _makespan_fast(best_perm, pt)
        
        # Try all permutations if small enough
        if n <= 8:
            for perm in permutations(range(n)):
                if time.time() > deadline:
                    break
                ms = _makespan_fast(list(perm), pt)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = list(perm)
        return best_perm
    
    # For larger instances, use metaheuristic approach
    best_overall = None
    best_overall_ms = float('inf')
    
    # Phase 1: NEH initialization
    neh = _neh_order(pt)
    best_overall = list(neh)
    best_overall_ms = _makespan_fast(best_overall, pt)
    
    # Phase 2: VND refinement
    if time.time() < deadline:
        vnd_result = _vnd(best_overall, pt, deadline)
        vnd_ms = _makespan_fast(vnd_result, pt)
        if vnd_ms < best_overall_ms:
            best_overall = vnd_result
            best_overall_ms = vnd_ms
    
    # Phase 3: Multiple restarts with GSA-like exploration
    n_particles = max(5, min(20, n))
    restarts_done = 0
    max_restarts = 50
    
    while time.time() < deadline and restarts_done < max_restarts:
        restarts_done += 1
        
        # Generate diverse starting points
        if restarts_done == 1:
            # Start from current best
            start_perm = list(best_overall)
        else:
            # Random perturbation of best
            start_perm = list(best_overall)
            n_swaps = random.randint(1, min(5, n - 1))
            for _ in range(n_swaps):
                i, j = random.sample(range(n), 2)
                start_perm[i], start_perm[j] = start_perm[j], start_perm[i]
        
        # GSA step
        gsa_perm, gsa_ms = _gsa_step(start_perm, pt, n_particles, deadline)
        
        if gsa_ms < best_overall_ms:
            best_overall = gsa_perm
            best_overall_ms = gsa_ms
        
        # VND on the GSA result
        if time.time() < deadline:
            vnd_result = _vnd(gsa_perm, pt, deadline)
            vnd_ms = _makespan_fast(vnd_result, pt)
            if vnd_ms < best_overall_ms:
                best_overall = vnd_result
                best_overall_ms = vnd_ms
    
    # Final VND polish
    if time.time() < deadline - 5.0:
        final_result = _vnd(best_overall, pt, deadline)
        final_ms = _makespan_fast(final_result, pt)
        if final_ms < best_overall_ms:
            best_overall = final_result
            best_overall_ms = final_ms
    
    return best_overall
# EVOLVE-BLOCK-END
