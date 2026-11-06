"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _calc_makespan(processing_times, order):
    """
    Calculate the makespan for a given job order efficiently.
    processing_times: np.ndarray [n_jobs, n_machines]
    order: list of job indices
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    
    # completion[m] is the time when machine m finishes the current job
    completion = np.zeros(n_machines, dtype=np.float64)
    
    for job in order:
        pt = processing_times[job]
        # Update completion times for each machine
        # C[m] = max(C[m], C[m-1]) + pt[m]
        # We need to be careful with in-place updates or use a temp variable
        prev_completion = 0.0
        for m in range(n_machines):
            # The machine cannot start before the previous machine finished the same job
            # AND before the machine finished the previous job
            start_time = max(completion[m], prev_completion)
            completion[m] = start_time + pt[m]
            prev_completion = completion[m]
            
    return float(completion[-1])

def _neh_initial(processing_times, rng=None, randomize=False):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    Sort jobs by total processing time descending.
    Insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if rng is None:
        rng = random.Random(42)
        
    # Calculate total processing time for each job
    total_times = processing_times.sum(axis=1)
    
    # Sort indices by total time descending
    order = list(np.argsort(-total_times))
    
    # Optional randomization for diversity
    if randomize and n_jobs > 2:
        # Swap a few adjacent elements to create diversity
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.5:
                order[i], order[i+1] = order[i+1], order[i]
                
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        # Try inserting job at each position in current seq
        for pos in range(len(seq) + 1):
            # Create trial sequence
            # For efficiency, we can just compute makespan of the trial
            # But creating lists is O(n). 
            # Trial: seq[:pos] + [job] + seq[pos:]
            # To avoid list creation overhead in tight loops, we could optimize,
            # but for n < 100, it's fine.
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _calc_makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        
    return seq

def _insertion_local_search(processing_times, seq, time_limit, deadline=None):
    """
    Local search using insertion moves.
    Remove a job, try inserting it at all other positions, keep best.
    """
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_ms = _calc_makespan(processing_times, seq)
    best_ms = current_ms
    best_seq = list(seq)
    
    start_time = time.time()
    
    # We iterate until no improvement or time limit
    while True:
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(n_jobs):
            job = best_seq[i]
            # Remove job
            seq_reduced = best_seq[:i] + best_seq[i+1:]
            
            best_pos = i
            best_insert_ms = best_ms
            
            # Try inserting at all positions in reduced sequence
            for pos in range(n_jobs - 1):
                # Trial: seq_reduced[:pos] + [job] + seq_reduced[pos:]
                # Optimization: If pos is far from i, we can break early if we know it's worse?
                # No, makespan is not monotonic with position.
                trial = seq_reduced[:pos] + [job] + seq_reduced[pos:]
                ms = _calc_makespan(processing_times, trial)
                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_pos = pos
                    
            if best_insert_ms < best_ms - 1e-9:
                # Apply best move
                best_seq = seq_reduced[:best_pos] + [job] + seq_reduced[best_pos:]
                best_ms = best_insert_ms
                improved = True
                
        if not improved:
            break
            
    return best_seq, best_ms

def _simulated_annealing(processing_times, seq, time_limit, deadline=None, rng=None):
    """
    Simulated Annealing to escape local minima.
    """
    if rng is None:
        rng = random.Random()
        
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_seq = list(seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    # Initial temperature: roughly 10% of makespan
    temperature = max(current_ms * 0.1, 1.0)
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() < deadline and temperature > 1e-5:
        # Generate a neighbor
        # Random swap of two jobs
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        if i == j:
            continue
            
        new_seq = list(current_seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_ms = _calc_makespan(processing_times, new_seq)
        
        delta = new_ms - current_ms
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_seq = new_seq
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
                
        temperature *= cooling_rate
        
    return best_seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    # Safety margin: 120s limit, leave 5s for safety
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    best_perm = None
    best_ms = float('inf')
    
    # Phase 1: Generate diverse initial solutions using NEH variants
    # We will run a few NEH variants to find a good starting point
    
    # Variant 1: Standard NEH
    if time.time() < deadline - 10.0:
        perm1 = _neh_initial(processing_times, rng, randomize=False)
        ms1 = _calc_makespan(processing_times, perm1)
        if ms1 < best_ms:
            best_ms = ms1
            best_perm = perm1
            
    # Variant 2: NEH with slight randomization
    if time.time() < deadline - 10.0:
        perm2 = _neh_initial(processing_times, rng, randomize=True)
        ms2 = _calc_makespan(processing_times, perm2)
        if ms2 < best_ms:
            best_ms = ms2
            best_perm = perm2
            
    # Variant 3: NEH with different random seed
    if time.time() < deadline - 10.0:
        rng3 = random.Random(12345)
        perm3 = _neh_initial(processing_times, rng3, randomize=True)
        ms3 = _calc_makespan(processing_times, perm3)
        if ms3 < best_ms:
            best_ms = ms3
            best_perm = perm3
            
    # Phase 2: Local Search (Insertion)
    # This is the most critical part for improvement
    if time.time() < deadline - 5.0 and best_perm is not None:
        ls_deadline = deadline - 2.0
        best_perm, best_ms = _insertion_local_search(processing_times, best_perm, time_limit=10.0, deadline=ls_deadline)
        
    # Phase 3: Simulated Annealing for diversification
    # If we have time left, use SA to try to jump out of local minima
    if time.time() < deadline - 2.0 and best_perm is not None:
        sa_deadline = deadline - 1.0
        rng_sa = random.Random(99999)
        sa_perm, sa_ms = _simulated_annealing(processing_times, best_perm, time_limit=5.0, deadline=sa_deadline, rng=rng_sa)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm
            
            # Polish the SA result with local search if time permits
            if time.time() < deadline - 0.5:
                best_perm, best_ms = _insertion_local_search(processing_times, best_perm, time_limit=2.0, deadline=deadline - 0.1)
                
    return best_perm

# EVOLVE-BLOCK-END
