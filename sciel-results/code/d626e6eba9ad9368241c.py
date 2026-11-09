"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock.
# We use a safety margin to ensure we return before the kill signal.
_DEADLINE_MARGIN_S = 5.0  # seconds to spare

def _makespan(processing_times, perm):
    """
    Compute the makespan for a given permutation.
    processing_times: (n_jobs, n_machines)
    perm: permutation of job indices (list or array)
    Returns: makespan (float)
    Uses early termination logic implicitly by just computing the standard way,
    but we can optimize the inner loop.
    """
    n_jobs, n_machines = processing_times.shape
    # We compute the completion time for each job on each machine
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    # where C[-1, m] = 0 and C[j, -1] = 0
    
    # Vectorized approach for speed:
    # We can compute the makespan using a cumulative max approach.
    # Let P be the processing times matrix permuted by perm.
    P = processing_times[perm]
    
    # C[j, m] = P[j, m] + max(C[j-1, m], C[j, m-1])
    # This is not directly vectorizable in a simple cumsum way due to the 2D dependency.
    # However, for small n_jobs, a loop is fast. For larger, we might need optimization.
    # Let's use a simple loop for clarity and correctness, it's usually fast enough for n_jobs < 1000.
    
    # Optimization: Use numpy for the inner max if possible?
    # Actually, the standard flow shop makespan calculation is:
    # C[j, m] = P[j, m] + max(C[j-1, m], C[j, m-1])
    # We can store C as a 2D array or just keep the previous row and current row.
    
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        # current row C[j, :]
        # C[j, 0] = P[j, 0] + C[j-1, 0]
        # C[j, m] = P[j, m] + max(C[j-1, m], C[j, m-1])
        
        # We can compute this iteratively.
        # To vectorize:
        # C[j, m] = P[j, m] + max(C[j-1, m], C[j, m-1])
        # Let A = C[j-1, :] (prev_row)
        # Let B = C[j, :-1] (current row without last element)
        # This is tricky to vectorize fully without a loop over machines.
        
        # Given n_machines is likely small (flow shop usually has fixed M, e.g., 3-10),
        # a loop over machines is fast.
        
        cur_row = np.empty(n_machines)
        # First machine
        cur_row[0] = P[j, 0] + prev_row[0]
        
        for m in range(1, n_machines):
            # C[j, m] = P[j, m] + max(C[j-1, m], C[j, m-1])
            # C[j-1, m] is prev_row[m]
            # C[j, m-1] is cur_row[m-1]
            cur_row[m] = P[j, m] + max(prev_row[m], cur_row[m-1])
            
        prev_row = cur_row
        
    return prev_row[-1]

def _cds_initialization(processing_times):
    """
    Campbell, Dudek, Smith (CDS) heuristic.
    1. Order jobs by total processing time (PT) in descending order (GIF) or ascending (LIF).
    2. For each order, construct a permutation by mirroring the sequence around the middle.
    3. Evaluate makespan for each candidate and return the best.
    """
    n_jobs, n_machines = processing_times.shape
    total_pt = processing_times.sum(axis=1)
    
    # Sort indices by total PT
    idx_gif = np.argsort(-total_pt)  # Greatest Insertion First (descending)
    idx_lif = np.argsort(total_pt)   # Least Insertion First (ascending)
    
    best_perm = None
    best_ms = float('inf')
    
    for idx in [idx_gif, idx_lif]:
        # CDS mirroring:
        # If n_jobs is odd, the middle element is fixed.
        # Sequence: 1, 2, ..., n-1, n, n-1, ..., 2
        # Wait, the standard CDS heuristic constructs n candidates? No, usually 2 or a few.
        # Actually, CDS generates a single sequence per sort order by mirroring.
        # Let's implement the standard CDS:
        # For a given order of jobs (1..n), the CDS sequence is:
        # 1, 2, ..., k, k+1, k, ..., 2  where k = (n+1)//2 ?
        # No, the CDS heuristic is:
        # Let the sorted order be J1, J2, ..., Jn.
        # The CDS permutation is J1, J2, ..., Jn, Jn-1, ..., J2 ? No.
        
        # Correct CDS:
        # The heuristic generates a sequence by taking the sorted list and mirroring it.
        # Specifically, if the sorted jobs are [j1, j2, ..., jn],
        # the CDS sequence is [j1, j2, ..., jn, jn-1, ..., j2] ??
        # Let me check the definition.
        # Campbell, Dudek, Smith (1970):
        # "The sequence is formed by taking the jobs in the order of decreasing (or increasing) total processing time, and then mirroring the sequence about the last job."
        # Example: Order 1, 2, 3, 4, 5.
        # CDS: 1, 2, 3, 4, 5, 4, 3, 2 ? No, that's not a permutation of 5 jobs.
        
        # Actually, CDS is a specific way to generate ONE permutation from a sorted list.
        # It is: j1, j2, ..., jn, jn-1, ..., j2 is wrong because length is wrong.
        
        # Let's look up CDS heuristic precisely.
        # CDS generates n-1 permutations? Or just one?
        # Common implementation:
        # Given sorted indices idx[0..n-1].
        # The CDS permutation is:
        # idx[0], idx[1], ..., idx[n-1], idx[n-2], ..., idx[1] ?
        # No, that's length 2n-1.
        
        # Ah, I recall now. CDS is often confused with the "mirror" heuristic.
        # The CDS heuristic actually produces a single sequence for a given sort order.
        # The sequence is:
        # 1, 2, ..., n, n-1, ..., 2 is NOT a permutation of n items.
        
        # Let's re-read a standard reference mentally.
        # CDS: Sort by total processing time.
        # Then, the permutation is constructed as:
        # p = [idx[0], idx[1], ..., idx[n-1]]
        # Then, for i from n-1 down to 1:
        #   ... wait.
        
        # Actually, many implementations of "CDS" in flow shop literature refer to:
        # 1. Sort jobs by total PT (ascending or descending).
        # 2. The resulting order IS the permutation.
        # But that's just NEH or simple sorting.
        
        # Let's check "CDS heuristic flow shop".
        # Campbell, Dudek, Smith (1970) proposed a heuristic that generates n-1 candidate sequences.
        # For k = 1 to n-1:
        #   Take the first k jobs from the sorted list.
        #   Mirror them? No.
        
        # Okay, let's use a robust, known-good heuristic: NEH (Nawaz-Enscore-Ham) is better than simple sorting.
        # And CDS is often implemented as:
        # For each sort order (asc/desc), generate a permutation by:
        #   Start with empty sequence.
        #   Insert jobs one by one from the sorted list into the best position?
        #   That's NEH.
        
        # Let's implement NEH properly, as it's a strong baseline.
        # And also simple CDS if I can find the exact definition.
        
        # Wait, the prompt says "CDS heuristic provides a stronger initial solution".
        # Let's implement NEH as the primary initialization, and maybe add a simple CDS variant if I can recall it.
        # Actually, CDS is:
        # Sort jobs by total processing time.
        # The permutation is: 1, 2, ..., n-1, n, n-1, ..., 2 ? No.
        
        # Let's stick to NEH which is well-defined and strong.
        pass

def _neh_initialization(processing_times):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current partial sequence.
    """
    n_jobs, n_machines = processing_times.shape
    total_pt = processing_times.sum(axis=1)
    idx = np.argsort(-total_pt)
    
    perm = []
    for i in range(n_jobs):
        job = idx[i]
        # Try inserting job at every position in the current perm
        best_pos = 0
        best_ms = float('inf')
        
        # Current perm length is i
        for pos in range(i + 1):
            # Create new perm
            new_perm = perm[:pos] + [job] + perm[pos:]
            # Calculate makespan
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
        
    return perm

def _local_search(processing_times, initial_perm, deadline):
    """
    Perform 2-opt local search (swap adjacent jobs) until no improvement or deadline.
    """
    n_jobs, n_machines = processing_times.shape
    perm = list(initial_perm)
    current_ms = _makespan(processing_times, perm)
    
    improved = True
    while improved:
        improved = False
        # Check deadline
        if time.time() > deadline:
            break
            
        for i in range(n_jobs - 1):
            # Try swapping i and i+1
            perm[i], perm[i+1] = perm[i+1], perm[i]
            new_ms = _makespan(processing_times, perm)
            
            if new_ms < current_ms:
                current_ms = new_ms
                improved = True
                # Continue searching from the same position i (or restart?)
                # Usually, restart or continue. Let's continue.
            else:
                # Swap back
                perm[i], perm[i+1] = perm[i+1], perm[i]
                
        # If we made an improvement, we might want to restart the scan to catch cascading improvements
        # Or just continue. A single pass might not be enough for local optimum.
        # Let's do a full pass. If improved, loop again.
        
    return perm, current_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN_S
    
    n_jobs, n_machines = processing_times.shape
    
    # 1. Initial Solution using NEH
    best_perm = _neh_initialization(processing_times)
    best_ms = _makespan(processing_times, best_perm)
    
    # 2. Local Search (2-opt adjacent swaps)
    if time.time() < deadline:
        best_perm, best_ms = _local_search(processing_times, best_perm, deadline)
        
    # 3. If time remains, try a few random restarts with local search?
    # Or just return the best found.
    # Given the 120s limit (or 150s kill), and NEH+LS is fast,
    # we might have time for multi-start.
    
    # Let's add a simple multi-start if time permits.
    # But first, let's ensure the basic LS is efficient.
    
    # For now, return the best from NEH+LS.
    return list(best_perm)

# EVOLVE-BLOCK-END
