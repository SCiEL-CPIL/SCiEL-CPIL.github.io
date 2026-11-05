"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def _makespan(pt, perm):
    """Compute makespan for a given permutation. O(n*m) in numpy."""
    if len(perm) == 0:
        return 0.0
    sub = pt[perm]
    n, m = sub.shape
    if m == 0:
        return 0.0
    # C[j, k] = completion time of job j on machine k
    # C[j, 0] = sum of first j+1 jobs on machine 0
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + sub[j, k]
    # We can compute this row by row, but vectorized across jobs for each machine step
    # Actually, the recurrence is sequential in both j and k, so we can't fully vectorize.
    # But we can vectorize over jobs for a fixed machine if we process machines sequentially.
    # Let's use a different approach: for each machine, the completion times depend on the previous machine.
    
    # More efficient: use the fact that for a fixed permutation, we can compute makespan in O(n*m)
    # with a simple loop over machines, and within each machine, we need the max of (prev machine completion, current machine cumulative).
    
    # Let's do it with a loop over machines, and use numpy for the cumulative part.
    # C on machine k for job j = max(C_{j, k-1}, C_{j-1, k}) + p_{j,k}
    # This is inherently sequential in j for fixed k.
    
    # Alternative: for small n, just do it in Python. For larger n, we can use a trick.
    # Actually, let's just do a straightforward O(n*m) computation.
    
    # For n up to ~100 and m up to ~10, this is 1000 operations, which is fast.
    # Let's use a numpy-based approach where possible.
    
    # C[j] tracks the completion time of job j on the current machine.
    # We process machines one by one.
    # For machine 0: C[j] = cumsum(sub[:, 0])
    # For machine k > 0: C[j] = max(C_prev[j], C[j-1]) + sub[j, k]
    # where C_prev is the completion on machine k-1, and C[j-1] is the completion of job j-1 on machine k.
    
    C_prev = np.cumsum(sub[:, 0])
    for k in range(1, m):
        C_curr = np.empty(n)
        C_curr[0] = C_prev[0] + sub[0, k]
        if n > 1:
            # For j >= 1: C_curr[j] = max(C_prev[j], C_curr[j-1]) + sub[j, k]
            # This is sequential, so we can't vectorize it directly.
            # But for small n, it's fine.
            for j in range(1, n):
                C_curr[j] = max(C_prev[j], C_curr[j-1]) + sub[j, k]
        C_prev = C_curr
    return C_prev[-1]


def _makespan_batch(pt, perms):
    """Compute makespans for a batch of permutations. perms shape (batch, n)."""
    batch_size = len(perms)
    result = np.empty(batch_size)
    for i in range(batch_size):
        result[i] = _makespan(pt, perms[i])
    return result


def _neh_initial(pt):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time,
    choosing the best insertion position at each step."""
    n, m = pt.shape
    order = np.argsort(-pt.sum(axis=1))
    perm = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(pt, np.array(trial))
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return np.array(perm)


def _ox_crossover(p1, p2):
    """Order Crossover (OX1)."""
    n = len(p1)
    if n <= 2:
        return p1.copy(), p2.copy()
    
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child1 = np.full(n, -1, dtype=int)
    child2 = np.full(n, -1, dtype=int)
    
    # Copy segment from p1 to child1, from p2 to child2
    child1[a:b+1] = p1[a:b+1]
    child2[a:b+1] = p2[a:b+1]
    
    # Fill the rest in order of appearance in p2 (for child1) and p1 (for child2)
    # Elements not in the segment, in the order they appear in the other parent
    seg1 = set(p1[a:b+1].tolist())
    seg2 = set(p2[a:b+1].tolist())
    
    # For child1: fill positions other than [a, b] with elements from p2 in order, skipping those in seg1
    fill1 = [x for x in p2 if x not in seg1]
    # For child2: fill positions other than [a, b] with elements from p1 in order, skipping those in seg2
    fill2 = [x for x in p1 if x not in seg2]
    
    # Fill child1
    pos = 0
    for i in range(n):
        if i < a or i > b:
            child1[i] = fill1[pos]
            pos += 1
    
    # Fill child2
    pos = 0
    for i in range(n):
        if i < a or i > b:
            child2[i] = fill2[pos]
            pos += 1
    
    return child1, child2


def _swap_mutation(perm, rate=0.1):
    """Swap mutation: swap two random positions with probability rate."""
    n = len(perm)
    if n < 2:
        return perm.copy()
    child = perm.copy()
    if np.random.random() < rate:
        i, j = np.random.choice(n, 2, replace=False)
        child[i], child[j] = child[j], child[i]
    return child


def _insert_mutation(perm, rate=0.1):
    """Insertion mutation: remove a job and reinsert it at a random position."""
    n = len(perm)
    if n < 2:
        return perm.copy()
    child = perm.copy()
    if np.random.random() < rate:
        i = np.random.choice(n)
        job = child[i]
        # Remove job
        child = np.delete(child, i)
        # Insert at random position
        j = np.random.choice(n)
        child = np.insert(child, j, job)
    return child


def _2opt_mutation(perm, rate=0.1):
    """2-opt style: reverse a segment."""
    n = len(perm)
    if n < 3:
        return perm.copy()
    child = perm.copy()
    if np.random.random() < rate:
        i, j = sorted(np.random.choice(n, 2, replace=False))
        child[i:j+1] = child[i:j+1][::-1]
    return child


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=float)
    n, m = pt.shape
    deadline = time.time() + 110.0  # 10s safety margin
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        # Try both permutations
        ms01 = _makespan(pt, np.array([0, 1]))
        ms10 = _makespan(pt, np.array([1, 0]))
        return [0, 1] if ms01 <= ms10 else [1, 0]
    
    # Initial population
    pop_size = min(200, max(50, n * 10))
    
    # Seed with NEH and random permutations
    neh_perm = _neh_initial(pt)
    pop = [neh_perm.copy()]
    
    # Add some bottleneck-machine-based initial solutions
    # Sort by processing time on the machine with highest total load
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    order_bottleneck = np.argsort(-pt[:, bottleneck])
    pop.append(order_bottleneck.copy())
    
    # Add random permutations
    while len(pop) < pop_size:
        pop.append(np.random.permutation(n))
    
    # Evaluate initial population
    best_perm = pop[0].copy()
    best_ms = _makespan(pt, best_perm)
    
    for p in pop[1:]:
        ms = _makespan(pt, p)
        if ms < best_ms:
            best_ms = ms
            best_perm = p.copy()
    
    # Genetic algorithm loop
    elite_size = max(5, pop_size // 10)
    tournament_size = 5
    crossover_rate = 0.8
    mutation_rates = (0.1, 0.1, 0.1)  # swap, insert, 2opt
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Tournament selection
        new_pop = []
        
        # Elitism: carry over best solutions
        pop_sorted = sorted(pop, key=lambda p: _makespan(pt, p))
        for p in pop_sorted[:elite_size]:
            new_pop.append(p.copy())
        
        # Generate offspring
        while len(new_pop) < pop_size and time.time() < deadline:
            # Tournament selection for parents
            idx1 = np.random.choice(pop_size, tournament_size, replace=False)
            idx2 = np.random.choice(pop_size, tournament_size, replace=False)
            p1 = min(pop[idx1], key=lambda p: _makespan(pt, p))
            p2 = min(pop[idx2], key=lambda p: _makespan(pt, p))
            
            if np.random.random() < crossover_rate:
                c1, c2 = _ox_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Apply mutations
            mtype = np.random.choice(3)
            if mtype == 0:
                c1 = _swap_mutation(c1, mutation_rates[0])
                c2 = _swap_mutation(c2, mutation_rates[0])
            elif mtype == 1:
                c1 = _insert_mutation(c1, mutation_rates[1])
                c2 = _insert_mutation(c2, mutation_rates[1])
            else:
                c1 = _2opt_mutation(c1, mutation_rates[2])
                c2 = _2opt_mutation(c2, mutation_rates[2])
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        pop = new_pop
        
        # Track best
        for p in pop:
            ms = _makespan(pt, p)
            if ms < best_ms:
                best_ms = ms
                best_perm = p.copy()
    
    return list(best_perm)
# EVOLVE-BLOCK-END
