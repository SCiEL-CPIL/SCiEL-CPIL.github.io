"""
Simulated Annealing with Or-opt moves for flow-shop permutation optimization.
Uses vectorized NumPy for fast makespan evaluation and a cooling schedule to
escape local minima, unlike the deterministic hill-climbing of the previous strategy.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_vec(perm, pt):
    """
    Compute makespan using vectorized operations.
    This is significantly faster than a pure Python loop for moderate n, m.
    """
    # perm: array of job indices
    # pt: (n_jobs, n_machines)
    
    # Extract processing times for the current permutation
    # p: (n_jobs, n_machines)
    p = pt[perm]
    
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    # We can compute this iteratively over machines or jobs.
    # Vectorizing over jobs is tricky due to dependency on C[j-1, m].
    # Vectorizing over machines is easy: C[j, :] depends on C[j-1, :] and C[j, 0..m-1].
    
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # Initialize C
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    # First job
    C[0, 0] = p[0, 0]
    if n_machines > 1:
        C[0, 1:] = np.cumsum(p[0, 1:])
        
    for j in range(1, n_jobs):
        # C[j, 0] = C[j-1, 0] + p[j, 0]
        C[j, 0] = C[j-1, 0] + p[j, 0]
        if n_machines > 1:
            # For m > 0:
            # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
            # This has a sequential dependency on m within the row.
            # However, we can unroll or use a simple loop for m.
            # Given n_machines is usually small (e.g., 5-20), a loop over m is fine.
            # Alternatively, we can observe that C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
            # Let's try to vectorize if possible.
            # Actually, the loop over j is the outer loop. The inner loop over m is small.
            for m in range(1, n_machines):
                C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
                
    return C[-1, -1]

def _neh_order(pt):
    """
    NEH heuristic: sort jobs by total processing time descending,
    insert each job at the position that minimizes makespan.
    """
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    perm = []
    for job in sorted_jobs:
        if not perm:
            perm.append(job)
            continue
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_vec(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return np.array(perm, dtype=int)

def _simulated_annealing(perm, pt, time_limit, deadline, seed):
    """
    Simulated Annealing with Or-opt moves.
    """
    rng = np.random.RandomState(seed)
    current_perm = perm.copy()
    current_ms = _makespan_vec(current_perm, pt)
    
    best_perm = current_perm.copy()
    best_ms = current_ms
    
    start_time = time.time()
    
    # Temperature schedule
    # Start high enough to accept bad moves, cool down to explore locally.
    # T0 depends on problem size.
    n_jobs = len(perm)
    T0 = 50.0  # Initial temperature
    T_end = 0.01
    # Number of iterations estimate? We don't know exactly.
    # Use time-based cooling.
    # alpha: cooling rate per second?
    # Let's use a fixed number of steps per "phase" or just iterate until time limit.
    
    # To make it robust, let's run a fixed number of iterations scaled by n_jobs
    # or just loop until deadline.
    
    # Or-opt move: remove a block of size 1, 2, or 3 and re-insert it.
    # We will sample random moves.
    
    while time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline - 0.1:
            break
            
        # Generate a random Or-opt move
        # 1. Choose block size
        block_size = rng.randint(1, 4)  # 1, 2, or 3
        if block_size > n_jobs:
            block_size = n_jobs
            
        # 2. Choose start position of the block
        if n_jobs - block_size <= 0:
            continue
        start = rng.randint(0, n_jobs - block_size)
        
        # Extract block
        block = current_perm[start:start+block_size].copy()
        
        # Remove block
        # New permutation without block
        if start == 0:
            remaining = current_perm[start+block_size:].copy()
        elif start + block_size == n_jobs:
            remaining = current_perm[:start].copy()
        else:
            remaining = np.concatenate((current_perm[:start], current_perm[start+block_size:]))
        
        # 3. Choose insertion position
        # Insert into remaining perm
        # Positions: 0 to len(remaining)
        ins_pos = rng.randint(0, len(remaining) + 1)
        
        # Construct new permutation
        if ins_pos == 0:
            new_perm = np.concatenate((block, remaining))
        elif ins_pos == len(remaining):
            new_perm = np.concatenate((remaining, block))
        else:
            new_perm = np.concatenate((remaining[:ins_pos], block, remaining[ins_pos:]))
            
        # Evaluate new makespan
        new_ms = _makespan_vec(new_perm, pt)
        
        # Metropolis acceptance criterion
        delta = new_ms - current_ms
        if delta < 0:
            # Accept improvement
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm.copy()
        else:
            # Accept worse with probability exp(-delta / T)
            # Need to update temperature.
            # Simple linear cooling based on time?
            elapsed = time.time() - start_time
            # Total time budget for this SA run
            total_budget = time_limit
            if total_budget <= 0:
                break
            progress = elapsed / total_budget
            if progress >= 1.0:
                break
            T = T0 * (T_end / T0) ** progress
            if T < T_end:
                T = T_end
            
            # Avoid division by zero or underflow
            if T > 1e-6:
                prob = np.exp(-delta / T)
                if rng.rand() < prob:
                    current_perm = new_perm
                    current_ms = new_ms
            # else: never accept

    return best_perm, best_ms

def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_total = time.time()
    # Safety margin: 150s limit, keep 5s buffer
    deadline = start_total + 145.0
    
    # 1. NEH initial solution
    neh_perm = _neh_order(processing_times)
    neh_ms = _makespan_vec(neh_perm, processing_times)
    
    best_perm = neh_perm
    best_ms = neh_ms
    
    # 2. Multi-start Simulated Annealing
    # We have ~140 seconds left.
    # Split time into multiple SA runs to escape local minima.
    # Let's do 5-10 runs depending on n_jobs.
    
    elapsed = time.time() - start_total
    time_left = deadline - elapsed
    
    if time_left > 5.0:
        # Determine number of restarts
        # More restarts for smaller problems? Or just fixed?
        # Let's allocate equal time to each restart.
        n_restarts = 5
        if n_jobs > 50:
            n_restarts = 3
        elif n_jobs < 10:
            n_restarts = 10
            
        time_per_restart = time_left / n_restarts
        
        for i in range(n_restarts):
            if time.time() > deadline - 1.0:
                break
                
            # Create a perturbed initial solution for SA
            # If i == 0, use NEH.
            # If i > 0, perturb NEH or previous best?
            # Better to start from NEH with random perturbations to explore different basins.
            
            if i == 0:
                initial = neh_perm.copy()
            else:
                initial = neh_perm.copy()
                # Random Or-opt perturbations
                rng_init = np.random.RandomState(123 + i)
                n_perturb = max(1, n_jobs // 5)
                for _ in range(n_perturb):
                    block_size = rng_init.randint(1, 4)
                    if block_size > n_jobs:
                        block_size = n_jobs
                    start = rng_init.randint(0, n_jobs - block_size)
                    block = initial[start:start+block_size].copy()
                    if start == 0:
                        remaining = initial[start+block_size:].copy()
                    elif start + block_size == n_jobs:
                        remaining = initial[:start].copy()
                    else:
                        remaining = np.concatenate((initial[:start], initial[start+block_size:]))
                    ins_pos = rng_init.randint(0, len(remaining) + 1)
                    if ins_pos == 0:
                        initial = np.concatenate((block, remaining))
                    elif ins_pos == len(remaining):
                        initial = np.concatenate((remaining, block))
                    else:
                        initial = np.concatenate((remaining[:ins_pos], block, remaining[ins_pos:]))
            
            # Run SA
            sa_perm, sa_ms = _simulated_annealing(initial, processing_times, time_per_restart, deadline, seed=42 + i)
            
            if sa_ms < best_ms:
                best_ms = sa_ms
                best_perm = sa_perm
                
            # Update NEH? No, keep original NEH for diversity, or update to best found?
            # Updating to best found can help if it's a good basin, but reduces diversity.
            # Let's stick to perturbing the original NEH to ensure we explore different regions.
            
    return best_perm.tolist()
# EVOLVE-BLOCK-END
