"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """
    Compute the makespan for a given permutation.
    O(n * m)
    """
    if len(perm) == 0:
        return 0.0
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # C[m] is the completion time of the last job processed on machine m
    # We iterate through jobs in the order of perm.
    # For each job, we update the completion times for all machines.
    # C_next[m] = max(C_prev[m], C_next[m-1]) + p[job, m]
    
    # Optimization: Use a loop for machines inside the job loop.
    # This is efficient enough for typical flow shop sizes (n < 1000).
    c_prev = np.zeros(m)
    for j in perm:
        c_curr = np.zeros(m)
        # Machine 0
        c_curr[0] = c_prev[0] + pt[j, 0]
        # Machines 1..m-1
        for k in range(1, m):
            # max(completion on prev machine for this job, completion on current machine for prev job)
            c_curr[k] = max(c_curr[k-1], c_prev[k]) + pt[j, k]
        c_prev = c_curr
        
    return c_prev[-1]

def _makespan_insertion_delta(perm, pt, pos_old, pos_new):
    """
    Calculate the new makespan if the job at pos_old is moved to pos_new.
    This is an O((pos_new - pos_old) * m) operation if we only update the affected segment.
    However, for simplicity and robustness within the time limit, 
    we will implement a fast full recalculation that is vectorized where possible,
    or a highly optimized loop.
    
    Actually, for the memetic algorithm, we will use a fast local search.
    To make local search fast, we need an efficient makespan calculator.
    The standard O(n*m) loop is the bottleneck.
    
    Let's try a different approach for the local search:
    We will maintain the completion time matrix C[n][m] for the current permutation.
    Then, an insertion move only requires recalculating the rows from min(pos_old, pos_new) to max(pos_old, pos_new).
    This is O(|delta_pos| * m).
    
    However, implementing the incremental update correctly is tricky.
    Given the 120s limit and the need for a robust solution, 
    let's stick to a very fast full makespan calculation using NumPy vectorization tricks 
    if possible, or just a tight Python loop.
    
    Let's optimize _makespan to be as fast as possible in pure Python/NumPy.
    """
    pass # Placeholder, we will use the optimized version below in the main logic

def _fast_makespan(perm, pt):
    """
    Optimized makespan calculation.
    """
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # Using a list of floats is often faster than numpy arrays for small m in Python loops
    # due to overhead. But let's try to keep it simple.
    c = [0.0] * m
    for j in perm:
        p = pt[j]
        # c[0] = c[0] + p[0]
        # c[k] = max(c[k-1], c[k]) + p[k]
        # This dependency chain prevents full vectorization across machines for a single job.
        # But we can unroll or just loop.
        prev_c = c[0]
        c[0] = prev_c + p[0]
        for k in range(1, m):
            # max(c[k-1], c[k])
            if c[k-1] > c[k]:
                c[k] = c[k-1] + p[k]
            else:
                c[k] = c[k] + p[k]
    return c[m-1]

def _neh_heuristic(processing_times, random_seed=None):
    """
    NEH heuristic with randomized tie-breaking or slight noise for diversity.
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()
        
    n, m = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    # To create diversity, we can add small noise to the totals before sorting
    noise = rng.uniform(0, 0.1 * totals.max(), size=n)
    sort_keys = totals + noise
    order = list(np.argsort(-sort_keys))
    
    current_perm = [order[0]]
    
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        
        # Try all positions
        # Optimization: We can break early if the makespan increases significantly? No, keep it simple.
        for pos in range(len(current_perm) + 1):
            # Construct new perm
            # Instead of slicing which creates copies, we can just calculate makespan on a view?
            # No, _fast_makespan takes a list.
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _fast_makespan(new_perm, processing_times)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm, best_mk

def _local_search_insertion(perm, pt, time_limit=5.0):
    """
    Local search using insertion moves.
    We try to move each job to a better position.
    """
    start_time = time.time()
    n = len(perm)
    m = pt.shape[1]
    current_mk = _fast_makespan(perm, pt)
    best_perm = perm[:]
    best_mk = current_mk
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate through each job
        for i in range(n):
            job = perm[i]
            # Try moving it to all other positions
            # Optimization: Only try positions that are significantly different? 
            # For now, try all j != i
            best_pos_i = i
            best_mk_i = current_mk
            
            for j in range(n):
                if i == j:
                    continue
                # Move job from i to j
                # If j > i, the segment i+1..j shifts left
                # If j < i, the segment j..i-1 shifts right
                
                # Construct new perm
                # This is O(n)
                new_perm = perm[:i] + perm[i+1:j] + [job] + perm[j:i] + perm[j+1:] if j < i else \
                           perm[:j] + [job] + perm[j:i] + perm[i+1:]
                
                # Actually, the list slicing logic above is a bit messy. Let's use a cleaner method.
                # Remove job at i
                temp_perm = perm[:i] + perm[i+1:]
                # Insert at j (in the new list context)
                # If j > i, the index j in the original list corresponds to j-1 in the new list?
                # It's safer to just build the list explicitly.
                
                # Let's stick to a simpler O(n) construction for clarity, 
                # but we need speed.
                # For n < 100, O(n^2 * m) is 100^2 * 10 = 100k ops, which is fast.
                
                # Re-evaluating the construction:
                if j > i:
                    new_perm = perm[:i] + perm[i+1:j+1] + [job] + perm[j+1:]
                else:
                    new_perm = perm[:j] + [job] + perm[j:i+1] + perm[i+1:]
                    
                mk = _fast_makespan(new_perm, pt)
                if mk < best_mk_i:
                    best_mk_i = mk
                    best_pos_i = j
                    best_perm_i = new_perm
            
            if best_pos_i != i:
                perm = best_perm_i
                current_mk = best_mk_i
                if current_mk < best_mk:
                    best_mk = current_mk
                    best_perm = perm[:]
                improved = True
                
    return best_perm, best_mk

def _ordered_crossover(parent1, parent2):
    """
    Ordered Crossover (OX1).
    """
    n = len(parent1)
    # Select a random slice
    i, j = random.sample(range(n), 2)
    if i > j:
        i, j = j, i
        
    # Copy the slice from parent1
    child = [None] * n
    child[i:j+1] = parent1[i:j+1]
    
    # Fill the rest from parent2, skipping those already in the slice
    fill_values = [x for x in parent2 if x not in parent1[i:j+1]]
    
    # Fill positions before i and after j
    # We need to fill in order of appearance in parent2
    k = 0
    for pos in range(n):
        if child[pos] is None:
            child[pos] = fill_values[k]
            k += 1
            
    return child

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate diverse initial population using NEH with noise
    pop_size = 20
    population = []
    for i in range(pop_size):
        perm, mk = _neh_heuristic(processing_times, random_seed=i)
        population.append((mk, perm))
        
    # Sort by makespan
    population.sort(key=lambda x: x[0])
    best_perm = population[0][1]
    best_mk = population[0][0]
    
    # 2. Memetic Phase: Local Search on top individuals
    # We will perform local search on the top K individuals
    k_local = 5
    for i in range(k_local):
        mk, perm = population[i]
        # Time budget per local search
        # If we have 100s left, and 5 individuals, give ~10s each?
        # Let's be conservative: 2s each initially, then more if time allows.
        ls_time = 2.0
        new_perm, new_mk = _local_search_insertion(perm, processing_times, time_limit=ls_time)
        population[i] = (new_mk, new_perm)
        
        if new_mk < best_mk:
            best_mk = new_mk
            best_perm = new_perm
            
    # Re-sort population
    population.sort(key=lambda x: x[0])
    
    # 3. Evolutionary Phase: Crossover and Local Search
    # Iterate until time runs out
    while time.time() < deadline:
        # Select parents (tournament selection)
        # Pick 2 random parents from top 10
        parent_idx1 = random.randint(0, min(9, len(population)-1))
        parent_idx2 = random.randint(0, min(9, len(population)-1))
        
        p1 = population[parent_idx1][1]
        p2 = population[parent_idx2][1]
        
        # Crossover
        child = _ordered_crossover(p1, p2)
        
        # Local Search on child
        # Time budget: 1s per child
        child_perm, child_mk = _local_search_insertion(child, processing_times, time_limit=1.0)
        
        # Replace worst individual in population
        # Find worst (last)
        worst_idx = len(population) - 1
        if child_mk < population[worst_idx][0]:
            population[worst_idx] = (child_mk, child_perm)
            population.sort(key=lambda x: x[0])
            
            if child_mk < best_mk:
                best_mk = child_mk
                best_perm = child_perm
                
    return list(best_perm)
# EVOLVE-BLOCK-END
