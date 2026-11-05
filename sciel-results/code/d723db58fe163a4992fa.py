"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan_vec(perm, pt):
    """
    Vectorized makespan calculation for a given permutation.
    pt: (n_jobs, n_machines) array of processing times.
    perm: (n_jobs,) array of job indices in order.
    Returns the makespan (completion time of the last job on the last machine).
    """
    if len(perm) == 0:
        return 0.0
    
    # Get processing times for the jobs in the order of perm
    # p is (n_jobs, n_machines)
    p = pt[perm]
    
    # Calculate cumulative sums along the machine axis for each job
    # cumsum_p is (n_jobs, n_machines)
    cumsum_p = np.cumsum(p, axis=1)
    
    # The completion time of job j on machine k is:
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
    # This can be computed iteratively, but we can use a trick with 
    # cumulative maxes if we reformulate, or just do a tight loop over jobs 
    # which is O(N*M) but with low constant factor if N is small.
    # However, for large N, a pure Python loop over N is slow.
    # Let's stick to a NumPy-based approach that is faster than pure Python loops.
    
    # Actually, the standard vectorized way for flow shop is tricky because of the 
    # dependency on the previous job's completion time on the same machine.
    # C[j, m] depends on C[j-1, m].
    # We can compute this in a loop over machines? No, C[j, m] depends on C[j, m-1].
    # It's a 2D recurrence.
    
    # A fast way is to use the fact that C[j, m] is the max over all paths.
    # But for speed, a simple loop over jobs with vectorized machine updates is often best.
    # Let's optimize the loop:
    
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # comp[m] is the completion time of the last processed job on machine m
    comp = np.zeros(n_machines)
    
    # p is (n_jobs, n_machines)
    # We iterate over jobs. For each job, we update comp.
    # comp_new[m] = max(comp[m], comp_new[m-1]) + p[j, m]
    # This is still O(N*M) operations, but inside a Python loop over N.
    # If N is 1000, 1000 loops is fast. If N is 10000, it might be slow.
    # Let's try to keep N reasonable or use a faster heuristic.
    
    # Alternative: If N is large, we might not be able to evaluate many candidates.
    # But the task implies synthetic instances, likely N < 1000.
    
    for j in range(n_jobs):
        # prev is the completion time of the previous machine for this job
        # comp[m] is the completion time of the previous job on this machine
        # We need to update comp in place carefully.
        # C[j, 0] = C[j-1, 0] + p[j, 0]
        # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
        
        # We can do this in a single pass if we store the previous comp[m]
        # before overwriting it.
        
        # Since p[j] is a vector, and comp is a vector.
        # Let's just do the loop. It's the most robust.
        
        # Optimization: If n_machines is small, this is very fast.
        # If n_jobs is large, we are limited by N.
        
        # Let's try a slightly optimized version:
        # We don't need to create a new array for each step.
        # We just update comp.
        
        # Note: comp[m] on the RHS must be the value from the previous job (j-1).
        # comp[m-1] on the RHS must be the value from the current job (j).
        
        # We can iterate backwards? No.
        # We can store prev_comp_m.
        
        # Actually, the simplest vectorized-ish way in Python for this specific recurrence
        # is hard to beat a tight loop unless N is huge.
        
        # Let's assume N is moderate (e.g., < 2000).
        pass

    # Re-implementing the tight loop properly here:
    comp = np.zeros(n_machines)
    for j in range(n_jobs):
        # p_j is the processing times for job j
        p_j = p[j]
        # We need to update comp
        # C_j_m = max(C_{j-1}_m, C_j_{m-1}) + p_j_m
        
        # Let's do it manually to avoid overhead of list conversions
        # C_j_0 = C_{j-1}_0 + p_j_0
        # C_j_m = max(C_{j-1}_m, C_j_{m-1}) + p_j_m
        
        # We can compute this in a loop over machines, which is small (M is usually small, e.g., 3-10)
        # So the outer loop is N, inner is M.
        
        # This is O(N*M).
        
        # Let's just use the standard loop.
        
        # To make it faster, we can use the fact that M is small.
        
        # Let's try to vectorize the inner loop? No, it's sequential.
        
        # Okay, let's stick to the loop but ensure it's as fast as possible.
        
        # Actually, there is a trick.
        # C[j, m] = max( max_{k<=m} (sum_{l=0}^k p[j, l]) + C[j-1, m-k-1] ) ... no that's complex.
        
        # Let's just do the loop.
        
        # Wait, I can use the previous implementation's logic which was correct.
        
        # Let's rewrite the function to be the fastest possible Python/NumPy hybrid.
        
        # If N is very large, we might need to prune. But let's assume standard sizes.
        
        # Let's use the loop.
        
        # One optimization: if M is 1, it's just sum.
        if n_machines == 1:
            comp[0] += p_j[0]
        else:
            # General case
            # We need to be careful with the dependency.
            # Let's use a temporary variable for the previous machine's completion time in the current job.
            prev_c = comp[0] # This is C[j-1, 0]
            # C[j, 0] = C[j-1, 0] + p[j, 0]
            comp[0] = prev_c + p_j[0]
            
            for m in range(1, n_machines):
                # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
                # C[j-1, m] is comp[m] (before update)
                # C[j, m-1] is comp[m-1] (after update)
                c_prev_job = comp[m]
                c_prev_machine = comp[m-1]
                comp[m] = max(c_prev_job, c_prev_machine) + p_j[m]
                
    return comp[-1]

def _makespan(perm, pt):
    """Wrapper for _makespan_vec to keep interface consistent if needed, but we'll use _makespan_vec."""
    return _makespan_vec(perm, pt)

def _neh(processing_times):
    """
    NEH heuristic:
    1. Sort jobs by decreasing total processing time.
    2. Insert each job into the best position in the current permutation.
    """
    n, m = processing_times.shape
    if n == 0:
        return np.array([], dtype=int)
    
    total_pt = processing_times.sum(axis=1)
    # Sort indices by decreasing total processing time
    order = np.argsort(-total_pt)
    
    perm = [order[0]]
    
    for k in range(1, n):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        
        # Try all positions
        # Optimization: We can limit the search space if n is large, but for now try all.
        # To speed up, we can early stop if we find a very good one? No, need exact best.
        
        # We can vectorize the evaluation of all positions?
        # Not easily because the makespan calculation is sequential.
        
        # However, we can use a heuristic to prune? No.
        
        # Let's just loop.
        # To make this faster, we can use the fact that we are inserting one job.
        
        for pos in range(len(perm) + 1):
            # Create trial permutation
            # Instead of creating a new list every time, we can compute makespan on the fly?
            # No, _makespan_vec takes a perm array.
            # Creating the array is O(N).
            
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_vec(np.array(trial, dtype=int), processing_times)
            
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        perm = perm[:best_pos] + [job] + perm[best_pos:]
        
    return np.array(perm, dtype=int)

def _insertion_ls(perm, pt, max_iterations=100):
    """
    Local search using insertions.
    Try moving each job to every other position.
    Accept the best improvement in each pass.
    """
    n = len(perm)
    if n < 2:
        return perm.copy()
    
    best_perm = perm.copy()
    best_ms = _makespan_vec(best_perm, pt)
    improved = True
    iters = 0
    
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        
        # Try all jobs
        for i in range(n):
            job = best_perm[i]
            
            # Remove job from position i
            # We need to evaluate insertion at all positions j != i
            
            # Optimization: We can skip j=i.
            # Also, symmetry? No.
            
            best_pos_for_job = -1
            best_ms_for_job = best_ms
            
            # Create a list without the job at i
            # base_perm = best_perm without index i
            
            # To avoid creating many arrays, we can just create the trial array.
            
            for j in range(n):
                if j == i:
                    continue
                
                # Construct trial permutation
                # Remove i, insert at j
                # Note: if j > i, the index shifts.
                
                # Let's build the array
                # It's O(N) to build.
                
                # We can optimize by pre-computing the "base" permutation without the job?
                # base = np.delete(best_perm, i)
                # Then insert at j.
                
                # Let's just do it simply first.
                
                trial = list(best_perm)
                # Remove i
                del trial[i]
                # Insert at j
                # Note: j is the index in the original array.
                # If j > i, the target index in the reduced array is j-1.
                # If j < i, the target index is j.
                
                # Wait, "insert at position j" usually means the job ends up at index j.
                # If we remove i, the length is N-1.
                # If we want the job to be at index j in the final array:
                # If j < i: insert at j.
                # If j > i: insert at j-1.
                # If j == i: skip.
                
                # Actually, let's just iterate over insertion positions in the reduced array.
                # The reduced array has length N-1.
                # Possible insertion indices: 0 to N-1.
                
                # Let's restart the inner loop logic.
                
                # base_perm = best_perm without job at i
                # We can compute this once outside the j loop?
                # Yes.
                
                # Let's refactor.
                
                pass
            
            # Let's refactor the inner loop to be more efficient.
            
            # Create base permutation without job i
            # base = np.concatenate([best_perm[:i], best_perm[i+1:]])
            
            # Now try inserting job at all positions 0..N-1 in base.
            
            base = np.concatenate([best_perm[:i], best_perm[i+1:]])
            
            for j in range(n): # 0 to N-1
                # Insert job at position j in base
                # trial = base[:j] + [job] + base[j:]
                
                # This is O(N) per j. Total O(N^2) per i. Total O(N^3) per pass.
                # This might be too slow for large N.
                
                # Optimization: 
                # We can compute the makespan more efficiently if we maintain the completion times?
                # But the permutation changes.
                
                # For now, let's assume N is small enough (e.g., < 100).
                # If N is 100, N^3 = 1,000,000 operations. Makespan is O(N*M).
                # 100 * 10 = 1000 ops per makespan.
                # 1,000,000 * 1000 = 1e9 ops. Too slow.
                
                # We need a faster local search.
                
                # Let's limit the search.
                # Only try insertions within a window?
                # Or just use 2-opt which is O(N^2) candidates, each O(N*M).
                # 2-opt is better for speed.
                
                # Let's switch back to 2-opt but with vectorized makespan.
                
                break # Break out of this inefficient inner loop logic for now
                
        # Let's implement a faster 2-opt or limited insertion.
        
        # Let's try a "Best Improvement" 2-opt.
        
        best_swap_i = -1
        best_swap_j = -1
        best_swap_ms = best_ms
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Swap i and j
                # We can compute makespan faster if we only recompute the affected part?
                # No, flow shop completion times propagate.
                
                # Let's just use _makespan_vec.
                
                # To speed up, we can use a heuristic to skip some swaps?
                # No.
                
                # Let's just do it.
                
                # Optimization: Use NumPy swap?
                # trial = best_perm.copy()
                # trial[i], trial[j] = trial[j], trial[i]
                
                # This is O(N) copy + O(N*M) makespan.
                
                # If N=100, 100*100/2 = 5000 swaps.
                # 5000 * (100 copy + 1000 makespan) = 5000 * 1100 = 5.5e6 ops.
                # This is fast!
                
                trial = best_perm.copy()
                trial[i], trial[j] = trial[j], trial[i]
                ms = _makespan_vec(trial, pt)
                
                if ms < best_swap_ms:
                    best_swap_ms = ms
                    best_swap_i = i
                    best_swap_j = j
        
        if best_swap_i != -1:
            best_perm[best_swap_i], best_perm[best_swap_j] = best_perm[best_swap_j], best_perm[best_swap_i]
            best_ms = best_swap_ms
            improved = True
            
    return best_perm

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    # Safety margin: 140s
    deadline = start_time + 140.0
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    # 1. Initialize with NEH
    best_perm = _neh(processing_times)
    best_ms = _makespan_vec(best_perm, processing_times)
    
    # 2. Local Search
    best_perm = _insertion_ls(best_perm, processing_times, max_iterations=20)
    best_ms = _makespan_vec(best_perm, processing_times)
    
    # 3. Diversification Loop
    # If we have time, try restarting from perturbed versions of the best solution.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Perturb the best solution
        perturbed = best_perm.copy()
        
        # Random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
        # Local search on perturbed solution
        candidate = _insertion_ls(perturbed, processing_times, max_iterations=5)
        candidate_ms = _makespan_vec(candidate, processing_times)
        
        if candidate_ms < best_ms:
            best_ms = candidate_ms
            best_perm = candidate.copy()
            
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return list(best_perm)

# EVOLVE-BLOCK-END
