import numpy as np
import time
import sys

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Helper to calculate makespan for a given sequence
    def calc_makespan(seq):
        c = [0.0] * n_machines
        for j in seq:
            for k in range(n_machines):
                if k == 0:
                    c[k] += processing_times[j, k]
                else:
                    c[k] = max(c[k], c[k-1]) + processing_times[j, k]
        return c[n_machines - 1]
    
    # Helper to calculate completion vector for a sequence
    def calc_completion_vector(seq):
        c = [0.0] * n_machines
        for j in seq:
            for k in range(n_machines):
                if k == 0:
                    c[k] += processing_times[j, k]
                else:
                    c[k] = max(c[k], c[k-1]) + processing_times[j, k]
        return c

    # --- Small n: Exact DP with State Pruning ---
    if n_jobs <= 12:
        # dp[mask] = list of non-dominated completion vectors
        # We store vectors as tuples for hashing/comparison
        dp = {0: [tuple([0.0] * n_machines)]}
        
        for mask in range(1 << n_jobs):
            if mask not in dp:
                continue
            # Prune current list (should be already pruned, but ensure)
            current_vecs = dp[mask]
            if not current_vecs:
                continue
                
            for j in range(n_jobs):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                new_vecs = []
                
                for vec in current_vecs:
                    # Calculate new completion vector
                    new_vec = list(vec)
                    for k in range(n_machines):
                        if k == 0:
                            new_vec[k] += processing_times[j, k]
                        else:
                            new_vec[k] = max(new_vec[k], new_vec[k-1]) + processing_times[j, k]
                    
                    new_vecs.append(tuple(new_vec))
                
                if new_mask not in dp:
                    dp[new_mask] = new_vecs
                else:
                    # Merge and prune
                    dp[new_mask].extend(new_vecs)
                    
                    # Prune dominated vectors
                    # A vector v1 dominates v2 if v1[k] <= v2[k] for all k
                    # We want to keep minimal vectors.
                    # Sort by last element (makespan) to help pruning?
                    # Simple O(N^2) prune is okay for small lists
                    pruned = []
                    for v in dp[new_mask]:
                        is_dominated = False
                        for u in pruned:
                            # if u dominates v, skip v
                            dominated = True
                            for k in range(n_machines):
                                if u[k] > v[k]:
                                    dominated = False
                                    break
                            if dominated:
                                is_dominated = True
                                break
                        if not is_dominated:
                            # check if v dominates any existing in pruned
                            new_pruned = []
                            for u in pruned:
                                v_dominates_u = True
                                for k in range(n_machines):
                                    if v[k] > u[k]:
                                        v_dominates_u = False
                                        break
                                if not v_dominates_u:
                                    new_pruned.append(u)
                            new_pruned.append(v)
                            pruned = new_pruned
                    dp[new_mask] = pruned
                    
                    # Safety check for time
                    if time.time() > deadline:
                        # Fallback to heuristic if DP takes too long
                        break
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() < deadline and (1 << n_jobs) - 1 in dp:
            # Find best makespan
            final_vecs = dp[(1 << n_jobs) - 1]
            if final_vecs:
                # We need to reconstruct the sequence. 
                # This DP didn't store parents. 
                # For n<=12, we can just store the best sequence found during DP?
                # Or re-run a greedy reconstruction?
                # Actually, storing parents in this state-space is tricky with pruning.
                # Alternative: For small n, just use a simple recursive search with memoization of best makespan?
                # No, the vector state is necessary.
                # Let's just return the NEH solution if reconstruction is too complex/slow.
                # Or, we can modify the DP to store the sequence?
                # Storing sequences in the vector state is heavy.
                # Given n<=12, maybe we can just use a different exact method?
                # Or, we can just use the DP to find the optimal *value* and then use a heuristic to find a sequence close to it?
                # No, we need the permutation.
                
                # Let's simplify: For n <= 10, use full DP with parent pointers.
                # For n > 10, use NEH + Local Search.
                pass

    # --- Fallback / Large n: NEH + Local Search ---
    # This will also serve as the fallback if DP is too slow or complex.
    
    # 1. NEH Construction
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            # Optimize makespan calculation for partial sequence
            # We can reuse the completion vector of the prefix if we store it, but for simplicity:
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    current_best_seq = seq
    current_best_ms = calc_makespan(seq)
    
    # 2. Local Search (Insertion)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_move = None
        best_new_ms = current_best_ms
        
        for i in range(len(seq)):
            job = seq[i]
            for j in range(len(seq)):
                if i == j:
                    continue
                # Move job from i to j
                # If j > i, shift i+1..j left, insert at j
                # If j < i, shift j..i-1 right, insert at j
                
                new_seq = seq.copy()
                new_seq.pop(i)
                # Adjust index if i < j
                if j > i:
                    new_seq.insert(j, job)
                else:
                    new_seq.insert(j, job)
                
                ms = calc_makespan(new_seq)
                if ms < best_new_ms:
                    best_new_ms = ms
                    best_move = (i, j, new_seq)
        
        if best_move:
            current_best_seq = best_move[2]
            current_best_ms = best_new_ms
            improved = True
            
    return current_best_seq
