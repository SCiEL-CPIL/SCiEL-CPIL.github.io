"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, p):
    """Compute makespan for a given permutation.
    
    p: (n_jobs, n_machines) array
    perm: array of job indices in order
    """
    n_jobs = len(perm)
    n_machines = p.shape[1]
    
    # Fast computation using cumulative sums per machine
    # completion[i][j] = max(completion[i-1][j], completion[i][j-1]) + p[j][perm[i]]
    # We can vectorize this somewhat
    
    # Build the processing times in permutation order
    p_perm = p[perm]  # (n_jobs, n_machines)
    
    # Iterative computation - this is inherently sequential per job
    # but we can use numpy for the inner machine loop
    completion = np.zeros(n_machines)
    for i in range(n_jobs):
        # completion[j] = max(completion[j], completion[j-1]) + p_perm[i][j]
        # This is a sequential dependency across machines
        for j in range(n_machines):
            if j == 0:
                completion[j] += p_perm[i][j]
            else:
                completion[j] = max(completion[j], completion[j-1]) + p_perm[i][j]
    
    return completion[-1]


def _makespan_fast(perm, p):
    """Faster makespan using numpy where possible."""
    p_perm = p[perm]
    n_jobs, n_machines = p_perm.shape
    
    if n_machines == 1:
        return p_perm.sum()
    
    # For small n_machines, the loop is fine
    # For larger, we can try to be smarter
    completion = np.zeros(n_machines)
    for i in range(n_jobs):
        row = p_perm[i]
        # Sequential update
        prev = 0.0
        for j in range(n_machines):
            val = completion[j] + row[j]
            if j > 0:
                val = max(val, prev)
            completion[j] = val
            prev = val
    
    return completion[-1]


def _neh_heuristic(p):
    """NEH heuristic for flow shop scheduling."""
    n_jobs = p.shape[0]
    n_machines = p.shape[1]
    
    # Sort by total processing time in descending order
    totals = p.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build the permutation using NEH insertion
    perm = [order[0]]
    for job in order[1:]:
        # Try all insertion positions
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(np.array(new_perm), p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return perm


def _adjacent_swap(perm, i):
    """Swap adjacent jobs at positions i and i+1."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _block_move(perm, i, j):
    """Move block from position i to position j (i < j).
    The block is just the single job at i moved to position j."""
    new_perm = list(perm)
    job = new_perm[i]
    del new_perm[i]
    new_perm.insert(j, job)
    return new_perm


def _reverse_block(perm, i, j):
    """Reverse the block from position i to position j (inclusive)."""
    new_perm = list(perm)
    new_perm[i:j+1] = reversed(new_perm[i:j+1])
    return new_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using VNS."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    if n_machines == 1:
        # Optimal is any order, but let's return sorted by processing time
        return list(np.argsort(-p.flatten()))
    
    # Initialize with NEH heuristic
    best_perm = np.array(_neh_heuristic(p))
    best_ms = _makespan_fast(best_perm, p)
    
    # Also try reverse NEH and a few random starts
    alt_starts = [
        np.argsort(-p.sum(axis=1)),  # Total processing time
        np.argsort(-p.sum(axis=0)[0] * p.sum(axis=1)),  # Weighted
    ]
    for _ in range(5):
        alt_starts.append(np.random.permutation(n_jobs))
    
    for start_perm in alt_starts:
        start_perm = np.array(start_perm)
        ms = _makespan_fast(start_perm, p)
        if ms < best_ms:
            best_ms = ms
            best_perm = start_perm.copy()
    
    # Variable Neighborhood Search
    # Neighborhoods:
    # N1: Adjacent swaps
    # N2: Block moves (move job from i to j)
    # N3: Block reversals (reverse segment i..j)
    
    current_perm = best_perm.copy()
    current_ms = best_ms
    
    # Tabu list to avoid cycling
    tabu_list = set()
    tabu_tenure = 10
    
    def make_tabu_key(perm):
        return tuple(perm)
    
    def is_tabu(perm):
        return make_tabu_key(perm) in tabu_list
    
    def add_to_tabu(perm):
        key = make_tabu_key(perm)
        tabu_list.add(key)
        if len(tabu_list) > tabu_tenure * 3:
            # Remove oldest entries (simple: just clear and rebuild)
            # For simplicity, we'll just keep a bounded set
            pass
    
    def local_search(perm, ms, time_limit):
        """Perform local search within the current neighborhood."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(perm)
            
            # Try all adjacent swaps
            for i in range(n - 1):
                if time.time() >= time_limit:
                    break
                new_perm = _adjacent_swap(perm, i)
                new_ms = _makespan_fast(np.array(new_perm), p)
                if new_ms < ms - 1e-9:
                    perm = np.array(new_perm)
                    ms = new_ms
                    improved = True
                    add_to_tabu(perm)
        
        return perm, ms
    
    def neighborhood_search(perm, ms, time_limit):
        """Search within a specific neighborhood type."""
        improved = True
        n = len(perm)
        
        while improved and time.time() < time_limit:
            improved = False
            
            if time.time() >= time_limit:
                break
            
            # Block moves: move job from i to j
            for i in range(n):
                if time.time() >= time_limit:
                    break
                for j in range(n):
                    if i == j:
                        continue
                    new_perm = _block_move(perm, i, j)
                    new_ms = _makespan_fast(np.array(new_perm), p)
                    if new_ms < ms - 1e-9:
                        perm = np.array(new_perm)
                        ms = new_ms
                        improved = True
                        add_to_tabu(perm)
                        break
                if improved:
                    break
        
        return perm, ms
    
    def reversal_search(perm, ms, time_limit):
        """Search using block reversals."""
        improved = True
        n = len(perm)
        
        while improved and time.time() < time_limit:
            improved = False
            
            if time.time() >= time_limit:
                break
            
            for i in range(n):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n):
                    new_perm = _reverse_block(perm, i, j)
                    new_ms = _makespan_fast(np.array(new_perm), p)
                    if new_ms < ms - 1e-9:
                        perm = np.array(new_perm)
                        ms = new_ms
                        improved = True
                        add_to_tabu(perm)
                        break
                if improved:
                    break
        
        return perm, ms
    
    # VNS main loop
    neighborhoods = [
        (local_search, "adjacent"),
        (neighborhood_search, "block_move"),
        (reversal_search, "reversal"),
    ]
    
    k = 0  # Current neighborhood index
    no_improvement_count = 0
    
    while time.time() < deadline:
        time_limit = min(time.time() + 10.0, deadline)
        
        # Apply local search in current neighborhood
        current_perm, current_ms = neighborhoods[k][0](
            current_perm, current_ms, time_limit
        )
        
        if current_ms < best_ms - 1e-9:
            best_ms = current_ms
            best_perm = current_perm.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # If no improvement, move to next neighborhood
        if no_improvement_count >= 2:
            k = (k + 1) % len(neighborhoods)
            no_improvement_count = 0
            
            # If we've cycled through all neighborhoods without improvement,
            # perturb the current solution
            if k == 0:
                # Random perturbation: swap a random pair
                if n_jobs >= 2:
                    i, j = np.random.choice(n_jobs, 2, replace=False)
                    new_perm = list(current_perm)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_perm = np.array(new_perm)
                    new_ms = _makespan_fast(new_perm, p)
                    
                    # Accept if not too bad or if it's better
                    if new_ms < current_ms * 1.2 or new_ms < best_ms:
                        current_perm = new_perm
                        current_ms = new_ms
        
        # Check time
        if time.time() >= deadline:
            break
    
    # Final polish: one more pass of adjacent swaps
    if time.time() < deadline - 5:
        final_perm, final_ms = local_search(best_perm, best_ms, deadline - 5)
        if final_ms < best_ms:
            best_ms = final_ms
            best_perm = final_perm
    
    return list(best_perm.astype(int))
# EVOLVE-BLOCK-END
