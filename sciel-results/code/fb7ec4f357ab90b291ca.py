"""Rewritten program for flow-shop permutation task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(pt, perm):
    """Compute makespan for a given permutation using DP."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    completion = [0.0] * n_machines
    for j in perm:
        for m in range(n_machines):
            if m == 0:
                completion[0] += pt[j, m]
            else:
                completion[m] = max(completion[m], completion[m-1]) + pt[j, m]
    return completion[-1]

def _make_completion_matrix(pt, perm):
    """Build the full completion time matrix C[j][m] for the given permutation.
    C[j][m] = completion time of job at position j on machine m.
    Returns shape (n_jobs, n_machines) array and the makespan.
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return np.zeros((n_jobs, n_machines)), 0.0
    C = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        job = perm[j]
        for m in range(n_machines):
            if m == 0:
                C[j, m] = (C[j-1, m] if j > 0 else 0.0) + pt[job, m]
            else:
                C[j, m] = max(C[j, m-1], C[j-1, m] if j > 0 else 0.0) + pt[job, m]
    return C, C[-1, -1]

def _delta_insertion_makespan(pt, perm, C, i, k):
    """Compute new makespan after moving job from position i to position k.
    C is the completion matrix for the current perm.
    Returns the new makespan, or None if k is same as i or i-1 (no change).
    
    We remove job at position i and insert at position k in the remaining sequence.
    If k > i, the remaining sequence has positions 0..n-2, and we insert at k.
    If k <= i, the remaining sequence has positions 0..n-2, and we insert at k.
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return None
    
    # Determine if this is a no-op
    # When we remove position i, the job at old position k (if k != i) shifts.
    # If we insert at position k in the remaining list:
    #   If k < i: the job moves forward (left)
    #   If k > i: the job moves backward (right), but in the remaining list of length n-1
    #   If k == i-1 or k == i: it's roughly the same position
    
    # Build new permutation
    remaining = perm[:i] + perm[i+1:]
    new_perm = remaining[:k] + [perm[i]] + remaining[k:]
    
    # Check if same permutation
    if new_perm == list(perm):
        return None
    
    # Recompute makespan for the new permutation
    # For efficiency, we only recompute from the affected region
    # But for correctness and simplicity, let's recompute the full C matrix
    # However, we can be smarter: the changes are localized
    
    # Actually, let's just recompute the full completion matrix for the new perm
    # This is O(n*m) which is acceptable for moderate n, m
    new_C = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        job = new_perm[j]
        for m in range(n_machines):
            if m == 0:
                new_C[j, m] = (new_C[j-1, m] if j > 0 else 0.0) + pt[job, m]
            else:
                new_C[j, m] = max(new_C[j, m-1], new_C[j-1, m] if j > 0 else 0.0) + pt[job, m]
    
    return new_C[-1, -1]

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH heuristic with local search.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 5 seconds margin before 150s limit
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Step 1: NEH heuristic for initial solution
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    init_order = list(np.argsort(-totals))
    
    best_perm = init_order[:]
    best_makespan = _makespan(pt, best_perm)
    
    # Step 2: Local search - insertion neighborhood with hill climbing
    def _local_search(perm, max_iter=None):
        """Hill climbing with insertion neighborhood."""
        current_perm = perm[:]
        current_C, current_makespan = _make_completion_matrix(pt, current_perm)
        improved = True
        iter_count = 0
        
        while improved:
            improved = False
            iter_count += 1
            if max_iter and iter_count > max_iter:
                break
            if time.time() > deadline:
                break
            
            # Try all insertions
            for i in range(n_jobs):
                if time.time() > deadline:
                    break
                # Try inserting job at position i into each position k in the remaining
                for k in range(n_jobs):
                    # Skip no-op moves
                    if k == i or k == i - 1:
                        continue
                    
                    new_makespan = _delta_insertion_makespan(pt, current_perm, current_C, i, k)
                    if new_makespan is not None and new_makespan < current_makespan:
                        # Apply the move
                        remaining = current_perm[:i] + current_perm[i+1:]
                        current_perm = remaining[:k] + [current_perm[i]] + remaining[k:]
                        current_C, current_makespan = _make_completion_matrix(pt, current_perm)
                        improved = True
                        break  # Restart scan from beginning
                if improved:
                    break
            
        return current_perm, current_makespan
    
    # Run local search on the NEH solution
    best_perm, best_makespan = _local_search(best_perm)
    
    # Step 3: Multi-start with different initial solutions
    candidates = []
    
    # 1. NEH (total processing time descending)
    candidates.append(list(np.argsort(-totals)))
    
    # 2. Sort by mean processing time
    means = pt.mean(axis=1)
    candidates.append(list(np.argsort(-means)))
    
    # 3. Sort by standard deviation
    stds = pt.std(axis=1)
    candidates.append(list(np.argsort(-stds)))
    
    # 4. Sort by first machine processing time
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 5. Sort by last machine processing time
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 6. Random permutations
    rng = np.random.RandomState(42)
    for _ in range(8):
        candidates.append(rng.permutation(n_jobs).tolist())
    
    # Evaluate each candidate with local search
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Skip if we already evaluated this candidate
        if cand == best_perm:
            continue
            
        perm, makespan = _local_search(cand, max_iter=3)
        if makespan < best_makespan:
            best_makespan = makespan
            best_perm = perm
    
    # Final polish: run local search on the best solution with more iterations
    if time.time() < deadline - 5:
        best_perm, best_makespan = _local_search(best_perm, max_iter=10)
    
    # Return the best permutation found
    return best_perm

# EVOLVE-BLOCK-END
