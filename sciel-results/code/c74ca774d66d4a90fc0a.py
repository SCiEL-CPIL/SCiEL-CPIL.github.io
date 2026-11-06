"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _fast_count_violations(mask, n=1000):
    """
    Count number of sums with > 2 representations (violation penalty).
    Uses convolution for O(N^2) but C-speed execution.
    """
    if not np.any(mask):
        return 0
    
    # Convolution of mask with itself gives count of ordered pairs (a,b) such that a+b=s
    # mask is 0-indexed, so index i corresponds to value i+1.
    # Sum s corresponds to index s-2 in the convolution result?
    # Let's trace:
    # mask[i]=1 means value i+1 is in set.
    # conv[k] = sum_{i+j=k} mask[i]mask[j]
    # If value a=i+1 and b=j+1 are in set, sum is a+b = i+j+2.
    # So conv[k] corresponds to sum s = k+2.
    # We need sums s from 2 to 2n.
    # k ranges from 0 to 2n-2.
    
    conv = np.convolve(mask, mask, mode='full')
    
    # conv[k] is count of pairs (i,j) with i+j=k.
    # Number of unordered pairs {a,b} with a<=b and a+b=s is:
    # R(s) = (conv[s-2] + (1 if s is even and s/2 is in set else 0)) / 2
    # Let's compute R(s) for all s efficiently.
    
    # s ranges from 2 to 2n.
    # k = s - 2 ranges from 0 to 2n-2.
    
    # diagonal_term[k] is 1 if the sum s=k+2 is formed by a pair (a,a) where a is in set.
    # a = s/2 = (k+2)/2.
    # This requires s to be even, so k+2 even => k even.
    # And a must be an integer in 1..n.
    # a = (k+2)/2.
    # mask index for a is a-1 = (k+2)/2 - 1 = k/2.
    # So if k is even, check mask[k/2].
    
    # We only care about sums s where R(s) > 2.
    # R(s) = (conv[s-2] + diag) / 2 > 2  =>  conv[s-2] + diag > 4.
    
    # Let's create a boolean array for valid sums.
    # We need to check all s from 2 to 2n.
    # k from 0 to 2n-2.
    
    # Efficiently compute diag for all k:
    # diag[k] = 1 if k is even and mask[k//2] is True, else 0.
    # Note: k//2 must be a valid index in mask (0 to n-1).
    # If k//2 >= n, then a > n, so a is not in set (since set is in 1..n).
    # So diag[k] = 0 for k//2 >= n.
    
    diag = np.zeros(len(conv), dtype=np.int32)
    # k even indices
    even_k = np.arange(0, len(conv), 2)
    # indices in mask
    mask_indices = even_k // 2
    # valid mask indices are < n
    valid_mask_idx = mask_indices < n
    # for those, diag[k] = mask[mask_indices]
    # We can't directly assign because even_k and mask_indices are arrays.
    # We can use np.where
    # But simpler:
    # Create a temp array for diag values at even_k
    diag_vals = np.zeros(len(even_k), dtype=np.int32)
    diag_vals[valid_mask_idx] = mask[mask_indices[valid_mask_idx]]
    # Now map back to diag array
    diag[even_k] = diag_vals
    
    # Now compute total count for each sum
    # total_count[k] = conv[k] + diag[k]
    # Violation if total_count[k] > 4
    
    total_counts = conv + diag
    violations = np.sum(total_counts > 4)
    
    return int(violations)

def _greedy_construct(n=1000):
    """Original greedy construction as baseline."""
    chosen = []
    counts = {}
    for x in range(1, n + 1):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
        if all(trial[s] <= 2 for s in new_sums):
            chosen.append(x)
            counts = trial
    return chosen

def _local_search(mask, n=1000, time_budget=10.0, seed=42):
    """Simulated annealing / local search to improve set size while maintaining validity."""
    rng = random.Random(seed)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Start from greedy solution
    greedy = _greedy_construct(n)
    mask = np.zeros(n, dtype=bool)
    for x in greedy:
        mask[x - 1] = True
    
    current_size = len(greedy)
    best_mask = mask.copy()
    best_size = current_size
    
    # Compute current violation count (should be 0 for valid set)
    current_violations = _fast_count_violations(mask, n)
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    attempts = 0
    no_improve_streak = 0
    
    while time.time() < deadline:
        attempts += 1
        if attempts % 1000 == 0:
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Randomly pick an operation: add, remove, or swap
        # Bias towards adding if we are valid, or swapping if we are stuck
        op = rng.random()
        
        if op < 0.3:
            # Try to add a random element
            candidates = np.nonzero(~mask)[0]
            if len(candidates) == 0:
                continue
            idx = candidates[rng.randint(0, len(candidates) - 1)]
            new_mask = mask.copy()
            new_mask[idx] = True
            new_violations = _fast_count_violations(new_mask, n)
            new_size = int(np.sum(new_mask))
            
            if new_violations == 0 and new_size > current_size:
                mask = new_mask
                current_size = new_size
                current_violations = 0
                if current_size > best_size:
                    best_mask = mask.copy()
                    best_size = current_size
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            elif new_violations > 0:
                delta_v = new_violations - current_violations
                if delta_v > 0:
                    prob = np.exp(-delta_v / max(T, 1e-10))
                    if rng.random() < prob:
                        mask = new_mask
                        current_violations = new_violations
                        current_size = new_size
                        no_improve_streak += 1
                        
        elif op < 0.5:
            # Try to remove a random element
            candidates = np.nonzero(mask)[0]
            if len(candidates) == 0:
                continue
            idx = candidates[rng.randint(0, len(candidates) - 1)]
            new_mask = mask.copy()
            new_mask[idx] = False
            new_violations = _fast_count_violations(new_mask, n)
            new_size = int(np.sum(new_mask))
            
            if new_violations < current_violations:
                mask = new_mask
                current_violations = new_violations
                current_size = new_size
                no_improve_streak += 1
            elif new_violations == current_violations and new_size < current_size:
                if rng.random() < 0.05: # Lower prob to keep size
                    mask = new_mask
                    current_violations = new_violations
                    current_size = new_size
                    no_improve_streak += 1
                    
        else:
            # Swap: remove one, try to add another
            candidates_in = np.nonzero(mask)[0]
            candidates_out = np.nonzero(~mask)[0]
            if len(candidates_in) == 0 or len(candidates_out) == 0:
                continue
            idx_in = candidates_in[rng.randint(0, len(candidates_in) - 1)]
            idx_out = candidates_out[rng.randint(0, len(candidates_out) - 1)]
            new_mask = mask.copy()
            new_mask[idx_in] = False
            new_mask[idx_out] = True
            new_violations = _fast_count_violations(new_mask, n)
            new_size = int(np.sum(new_mask))
            
            if new_violations == 0 and new_size > current_size:
                mask = new_mask
                current_size = new_size
                current_violations = 0
                if current_size > best_size:
                    best_mask = mask.copy()
                    best_size = current_size
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            elif new_violations == 0 and new_size == current_size:
                if rng.random() < 0.1:
                    mask = new_mask
                    current_violations = 0
                    no_improve_streak += 1
            elif new_violations > 0:
                delta_v = new_violations - current_violations
                if delta_v > 0:
                    prob = np.exp(-delta_v / max(T, 1e-10))
                    if rng.random() < prob:
                        mask = new_mask
                        current_violations = new_violations
                        current_size = new_size
                        no_improve_streak += 1
    
    # Return the best valid mask found
    best_violations = _fast_count_violations(best_mask, n)
    if best_violations > 0:
        # Fallback to greedy if best is invalid
        return _greedy_construct(n)
    return [i + 1 for i in np.nonzero(best_mask)[0]]

def construct():
    start_time = time.time()
    # Hard deadline: 115 seconds to be safe
    total_budget = 115.0
    
    # Phase 1: Greedy baseline
    greedy_result = _greedy_construct(1000)
    best_result = greedy_result
    best_size = len(greedy_result)
    
    # Phase 2: Local search with multiple restarts
    # With vectorized counting, each step is fast.
    # We can run more restarts or longer runs.
    # Let's do restarts of 5 seconds each, or until time runs out.
    
    elapsed = time.time() - start_time
    # Reserve 5 seconds for final verification and return
    search_deadline = start_time + total_budget - 5.0
    
    r = 0
    while time.time() < search_deadline:
        r += 1
        seed = 42 + r * 137
        # Allocate time for this restart
        time_left = search_deadline - time.time()
        if time_left < 1.0:
            break
        
        # Use most of the remaining time for one long run, or split?
        # Splitting allows exploring different basins.
        # Let's do runs of max 5 seconds, or remaining time if less.
        time_per_restart = min(5.0, time_left)
        
        result = _local_search(np.zeros(1000, dtype=bool), n=1000, 
                               time_budget=time_per_restart, seed=seed)
        
        if len(result) > best_size:
            # Verify validity
            counts = {}
            valid = True
            for i in range(len(result)):
                for j in range(i, len(result)):
                    s = result[i] + result[j]
                    counts[s] = counts.get(s, 0) + 1
                    if counts[s] > 2:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_result = result
                best_size = len(result)
    
    # Final verification
    counts = {}
    valid = True
    for i in range(len(best_result)):
        for j in range(i, len(best_result)):
            s = best_result[i] + best_result[j]
            counts[s] = counts.get(s, 0) + 1
            if counts[s] > 2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return greedy_result
    
    return best_result
# EVOLVE-BLOCK-END
