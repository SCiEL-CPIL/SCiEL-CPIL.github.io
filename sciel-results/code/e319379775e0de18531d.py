import time
import random
import numpy as np
import scipy.sparse as sp

def construct():
    start_time = time.time()
    time_limit = 58.0 # Safety margin
    
    N = 256
    
    # 1. Precompute APs and reverse mapping (index -> list of AP indices)
    # We need to track sums for each AP.
    # ap_indices[r] = list of column indices in AP r
    
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1: # Include singletons as per task description
                ap_indices.append(indices)
                
    num_aps = len(ap_indices)
    
    # Reverse mapping: for each position i in 0..N-1, which APs contain it?
    # This is crucial for fast delta updates.
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)
            
    # Precompute the length of each AP (for potential tie-breaking or logic, though not strictly needed for sum update)
    ap_lengths = [len(ind) for ind in ap_indices]
    
    def evaluate_sequence(s):
        """
        Calculate the max absolute sum for all APs given a sequence s.
        s is a numpy array of -1/1.
        Returns max_abs_sum.
        """
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            # Sum of s[i] for i in indices
            # Using numpy sum on slice/list is fast enough for initialization
            sums[r] = np.sum(s[indices])
        return np.max(np.abs(sums))

    def greedy_optimize(s_init, time_budget):
        """
        Perform Greedy Coordinate Descent.
        s_init: numpy array of -1/1
        time_budget: seconds allowed for this run
        Returns: best_s, best_score
        """
        s = s_init.copy()
        
        # Initialize sums
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
            
        current_max = np.max(np.abs(sums))
        # Track which APs are currently at the max? 
        # To speed up, we can just recompute max when needed, or maintain a count of how many APs have abs(sum) == current_max.
        # For simplicity and robustness, let's recompute max only when we accept a move that might lower it, 
        # or just keep a running max. 
        # Actually, recomputing max over 100k elements is fast (~0.1ms). 
        # The bottleneck was the dot product. Now we update specific sums.
        
        best_s = s.copy()
        best_score = current_max
        
        rng = random.Random(123)
        
        # Precompute list of indices to iterate over
        indices_list = list(range(N))
        
        while time.time() - start_time < time_budget:
            # Randomize order to escape symmetry/local traps slightly
            rng.shuffle(indices_list)
            
            improved = False
            for i in indices_list:
                # Check time frequently
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                # Flipping i changes sum of every AP containing i by -2*old_val
                # new_sum = old_sum - 2*old_val
                delta_val = -2 * old_val
                
                # Calculate new max if we flip
                # We need to know if this flip reduces the global max.
                # Let's simulate the flip on a copy of sums? No, that's expensive if we copy 100k array.
                # Instead, we can calculate the potential new max.
                
                # Optimization: 
                # 1. Identify APs containing i.
                # 2. Update their sums in a temporary view or just calculate the change.
                
                # Let's do a "dry run" to see if it improves.
                # We need to know the new max.
                # The new max is either:
                # a) The max of APs NOT containing i (unchanged)
                # b) The max of APs containing i (updated)
                
                # Let's store the current max.
                # If we flip, the sums of APs in aps_containing[i] change.
                # Let's calculate the new sums for those APs.
                
                # To avoid copying the whole sums array, we can track the max of the "unchanged" part.
                # This is hard to do efficiently without a heap or similar structure.
                
                # Alternative: Just update the sums array in-place, check new max, and revert if worse.
                # Updating sums is O(count(i)). Checking max is O(num_aps).
                # num_aps ~ 100k. O(100k) per step is ~0.1ms.
                # 1000 steps/sec -> 55s = 55k steps.
                # This is feasible.
                
                # Let's try the in-place update/revert approach.
                
                # Save old sums for affected APs
                affected_aps = aps_containing[i]
                old_sums = sums[affected_aps].copy()
                
                # Apply flip
                sums[affected_aps] += delta_val
                s[i] = -old_val
                
                new_max = np.max(np.abs(sums))
                
                if new_max < current_max:
                    # Strict improvement
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                    improved = True
                elif new_max == current_max:
                    # Tie. Use secondary metric: sum of squares?
                    # Or just accept to explore?
                    # Let's accept if it reduces the sum of squares of the sums (minimizes variance)
                    # This requires calculating sum of squares, which is O(num_aps).
                    # Let's skip secondary metric for speed, or use a cheaper one.
                    # Cheaper: if we flip, do we reduce the number of APs at max?
                    # Let's just accept ties with some probability to allow movement.
                    if rng.random() < 0.1: # 10% chance to accept tie
                        improved = True
                    else:
                        # Revert
                        sums[affected_aps] = old_sums
                        s[i] = old_val
                else:
                    # Worse
                    sums[affected_aps] = old_sums
                    s[i] = old_val
            
            if not improved:
                # Local minimum found. 
                # Break to restart? Or try a random perturbation?
                # Since we have a time budget, breaking allows the outer loop to restart.
                break
                
        return best_s, best_score

    # Main Loop: Multiple Restarts
    best_global_s = None
    best_global_score = float('inf')
    
    # We have ~58s. 
    # Each greedy run might take variable time.
    # Let's try a few restarts.
    
    rng_global = random.Random(42)
    
    # First run: Random init
    # We can use a simple heuristic for init: Thue-Morse?
    # Thue-Morse sequence: t(n) = parity of number of 1s in binary expansion of n.
    # Signs: +1 if t(n) even, -1 if odd? Or vice versa.
    # Let's try Thue-Morse as one of the inits, as it has low discrepancy for powers of 2.
    
    def thue_morse(N):
        seq = []
        for i in range(N):
            # Count bits
            count = bin(i).count('1')
            if count % 2 == 0:
                seq.append(1)
            else:
                seq.append(-1)
        return np.array(seq, dtype=np.int8)
    
    # Init 1: Thue-Morse
    s_init = thue_Morse(N) if 'thue_Morse' in locals() else thue_morse(N)
    
    # Let's run a few restarts.
    # Estimate time per restart. If a restart finishes quickly (local min), we can do more.
    # If it runs out of time, we stop.
    
    restarts = 0
    while time.time() - start_time < time_limit:
        if restarts == 0:
            s_init = thue_morse(N)
        elif restarts == 1:
            # Random
            s_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        else:
            # Random
            s_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            
        # Allocate remaining time, but leave a small buffer for the last potential restart or just use all
        # If we use all time, we might not have time to check the next restart condition.
        # Let's allocate 80% of remaining time to the current run, keeping 20% buffer.
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
        run_budget = remaining * 0.8
        
        s_best_run, score_best_run = greedy_optimize(s_init, run_budget)
        
        if score_best_run < best_global_score:
            best_global_score = score_best_run
            best_global_s = s_best_run
            
        restarts += 1
        
        # If we found a perfect solution (discrepancy 1 is theoretical min for N>1?), stop.
        # Min possible is 1 (singletons).
        if best_global_score <= 1:
            break

    if best_global_s is None:
        # Fallback
        best_global_s = thue_morse(N)
        
    return [int(x) for x in best_global_s]
