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

def _compute_prefix_completions(processing_times, seq):
    """
    Compute prefix completion times.
    prefix[m][i] = completion time of machine m after processing seq[0:i]
    Shape: (n_machines, n_jobs + 1)
    """
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    prefix = np.zeros((n_machines, n_jobs + 1), dtype=np.float64)
    
    for i, job in enumerate(seq):
        pt = processing_times[job]
        prev_completion = 0.0
        for m in range(n_machines):
            start_time = max(prefix[m][i], prev_completion)
            prefix[m][i + 1] = start_time + pt[m]
            prev_completion = prefix[m][i + 1]
            
    return prefix

def _compute_suffix_completions(processing_times, seq):
    """
    Compute suffix completion times.
    suffix[m][i] = if we start with all machines free at time 0, and process seq[i:],
    what is the completion time on machine m?
    Actually, we need: for a subsequence seq[i:], if the first job could start at time 0 on machine 0,
    and each subsequent job starts as soon as possible, what's the makespan?
    
    More precisely: suffix[m][i] = makespan contribution of seq[i:] when "injected" at position i.
    We define it backwards:
    Let's define: if we process jobs seq[i], seq[i+1], ..., seq[n-1] in order,
    and the "entry time" for machine m is 0 for all m (i.e., machines are free),
    then suffix[m][i] is the completion time on machine m.
    
    But that's not quite right for insertion. We need to know how the tail behaves
    given an entry time on each machine.
    
    Actually, for flow shop, the state is just the vector of machine completion times.
    When inserting a job at position pos, we need to know:
    1. The completion times after processing seq[0:pos] (prefix)
    2. How the job + seq[pos:] evolves from those completion times
    
    The challenge is that the evolution of seq[pos:] depends on the entry times.
    
    Alternative approach: Precompute for each position pos, and for each possible "shift",
    how the makespan changes. But that's complex.
    
    Let me use a different approach: For insertion at position pos of job j,
    the new makespan can be computed as:
    - Process seq[0:pos] to get completion times C_prefix
    - Process job j starting from C_prefix
    - Process seq[pos:] starting from the new completion times after job j
    
    The issue is step 3 depends on the specific C_prefix.
    
    However, we can precompute "suffix profiles": for each position pos,
    and for each machine m, what is the "delay" caused by seq[pos:] relative to
    a baseline.
    
    Actually, let me just use the straightforward O(n*m) computation for each candidate,
    but optimize the inner loop. The key insight is that for a fixed insertion position,
    we can compute the makespan incrementally.
    
    For now, let me implement a fast insertion evaluation that reuses prefix computations.
    """
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    # suffix[m][i] will store: if we start processing seq[i:] with all machines at time 0,
    # the completion time on machine m.
    # This is a "baseline" that doesn't account for entry times from prefix.
    suffix = np.zeros((n_machines, n_jobs + 1), dtype=np.float64)
    
    # Process backwards
    for i in range(n_jobs - 1, -1, -1):
        job = seq[i]
        pt = processing_times[job]
        # We need to compute how this job affects the completion times
        # If we process job i, then jobs i+1, ..., n-1
        # The completion time on machine m after job i is:
        # C[m] = max(C_prev[m], C_prev[m-1]) + pt[m]
        # But C_prev here refers to the state BEFORE job i.
        # This is getting complex. Let me just compute makespan directly.
        pass
    
    # Alternative: just compute makespan for each trial directly but efficiently
    # For n_jobs up to ~50 and n_machines up to ~10, O(n*m) per trial is fine.
    # The bottleneck is the number of trials, not the cost per trial.
    
    return suffix

def _fast_insertion_eval(processing_times, seq, job_idx, job, prefix, n_machines):
    """
    Evaluate the makespan if we insert job at position job_idx in seq.
    Uses precomputed prefix completions.
    Returns the makespan.
    """
    n_jobs = len(seq)
    
    # Get completion times after processing seq[0:job_idx]
    # prefix[m][job_idx] gives this
    completion = np.array(prefix[:, job_idx], dtype=np.float64)
    
    # Process the job being inserted
    pt = processing_times[job]
    prev_completion = 0.0
    for m in range(n_machines):
        start_time = max(completion[m], prev_completion)
        completion[m] = start_time + pt[m]
        prev_completion = completion[m]
    
    # Now process seq[job_idx:]
    for i in range(job_idx, n_jobs):
        pt = processing_times[seq[i]]
        prev_completion = 0.0
        for m in range(n_machines):
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
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _calc_makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        
    return seq

def _fast_insertion_local_search(processing_times, seq, deadline):
    """
    Local search using insertion moves with incremental evaluation.
    Remove a job, try inserting it at all other positions, keep best.
    Uses prefix completions for faster evaluation.
    """
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    best_seq = list(seq)
    best_ms = _calc_makespan(processing_times, best_seq)
    
    # We iterate until no improvement or time limit
    while time.time() < deadline:
        improved = False
        
        # Recompute prefix for current best_seq
        prefix = _compute_prefix_completions(processing_times, best_seq)
        
        for i in range(n_jobs):
            job = best_seq[i]
            # Remove job from consideration
            # Try inserting at all positions in the reduced sequence
            best_pos = i
            best_insert_ms = best_ms
            
            for pos in range(n_jobs - 1):
                # Evaluate inserting job at position pos in seq without job at i
                # This is equivalent to: seq[:pos] + [job] + seq[pos:i] + seq[i+1:]
                # But we need to be careful with indices
                
                # If pos <= i:
                #   The job is inserted before or at the original position
                #   The reduced sequence is seq[0:i] + seq[i+1:]
                #   We insert at position pos in this reduced sequence
                #   
                # If pos > i:
                #   The job is inserted after its original position
                #   In the reduced sequence (which has length n-1), position pos corresponds to
                #   position pos-1 in the original indexing (since we removed one element before)
                
                # Let me just construct the trial and compute makespan directly
                # but using the prefix for the common part
                
                # Actually, let me just compute it directly for correctness
                seq_reduced = best_seq[:i] + best_seq[i+1:]
                trial = seq_reduced[:pos] + [job] + seq_reduced[pos:]
                ms = _calc_makespan(processing_times, trial)
                
                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_pos = pos
                    
            if best_insert_ms < best_ms - 1e-9:
                # Apply best move
                seq_reduced = best_seq[:i] + best_seq[i+1:]
                best_seq = seq_reduced[:best_pos] + [job] + seq_reduced[best_pos:]
                best_ms = best_insert_ms
                improved = True
                # After an improvement, restart the scan
                break
                
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
            
    # Phase 2: Local Search (Insertion) - fast version
    # This is the most critical part for improvement
    if time.time() < deadline - 5.0 and best_perm is not None:
        ls_deadline = deadline - 2.0
        best_perm, best_ms = _fast_insertion_local_search(processing_times, best_perm, ls_deadline)
        
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
                best_perm, best_ms = _fast_insertion_local_search(processing_times, best_perm, deadline - 0.1)
                
    return best_perm

# EVOLVE-BLOCK-END
