"""Ternary covering code via Simulated Annealing on codeword population."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    
    Strategy:
    1. Precompute Hamming balls as boolean masks.
    2. Initialize a random set of codewords.
    3. Use Simulated Annealing to minimize the number of codewords while ensuring full coverage.
       - State: List of codeword indices.
       - Moves: Add, Remove, Swap.
       - Cost: lambda * (uncovered_count) + len(code).
       - Acceptance: Metropolis criterion based on temperature.
    4. Return the best valid code found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    N = q ** n
    if N <= 0:
        return []
    
    # Precompute all words as integer indices 0..N-1
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q
            
    # Compute Hamming balls: ball_mask[c] is a boolean array of size N
    # ball_mask[c, w] is True if dist(c, w) <= radius
    # For efficiency, we compute this on the fly or precompute if N is small enough.
    # N=2187, N^2 ~ 4.7M. This is small.
    hamming = np.zeros((N, N), dtype=np.int8)
    for k in range(n):
        col = all_words[:, k].astype(np.int8)
        diff = (col[:, None] != col[None, :]).astype(np.int8)
        hamming += diff
        
    ball_mask = (hamming <= radius)
    del hamming
    
    # Helper: Calculate coverage and cost
    # To speed up SA, we maintain a 'coverage_count' array where coverage_count[w] is the number of codewords covering w.
    # Initial state: Random selection of ~200 words.
    
    # Greedy initialization to get a valid start
    uncovered = np.ones(N, dtype=bool)
    code_set = []
    coverage_count = np.zeros(N, dtype=np.int32)
    
    while np.any(uncovered):
        # Find codeword covering most uncovered
        # new_cov[c] = number of uncovered words covered by c
        # We can approximate this by summing ball_mask[c] & uncovered
        # To speed up, we just pick a random one from uncovered? No, greedy is better for init.
        # But N=2187, loop is fine.
        best_idx = -1
        max_cov = 0
        # Vectorized greedy step is expensive if done naively in python loop.
        # Let's just pick a random word from uncovered and find a center? 
        # Or simply: pick random word w from uncovered, pick random center c that covers it?
        # Actually, standard greedy:
        # new_cov = np.sum(ball_mask & uncovered[None, :], axis=1)
        # This is O(N^2) per step. 200 steps * 2187^2 ~ 1e9 ops. Too slow in python?
        # Numpy vectorization handles it. 2187*2187 is ~4.7M bools. Summing along axis 1 is fast.
        
        new_cov = np.sum(ball_mask & uncovered[None, :], axis=1)
        best_idx = np.argmax(new_cov)
        
        if new_cov[best_idx] == 0:
             # Should not happen if covering exists, but fallback
             break
             
        code_set.append(int(best_idx))
        # Update coverage count
        coverage_count[ball_mask[best_idx]] += 1
        # Update uncovered
        uncovered[ball_mask[best_idx]] = False
        
        if time.time() > deadline - 5:
            break
            
    # Ensure valid
    if not np.all(coverage_count > 0):
         # Fallback if greedy failed (unlikely)
         return [list(all_words[i]) for i in code_set] if code_set else []

    best_code = code_set.copy()
    best_cost = len(best_code) # Since it's valid, cost is just size. 
    # For SA, we define cost = len(code) + 1000 * uncovered_count
    
    current_code = best_code.copy()
    current_coverage_count = coverage_count.copy()
    
    # SA Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    lambda_uncovered = 1000.0
    
    # Precompute neighbors? No, just random moves.
    
    # Helper to get current cost
    def get_cost(code, cov_count):
        uncovered_count = np.sum(cov_count == 0)
        return len(code) + lambda_uncovered * uncovered_count

    current_cost = get_cost(current_code, current_coverage_count)
    
    # Main SA Loop
    while time.time() < deadline - 5 and T > min_T:
        
        # Generate a move
        move_type = random.random()
        
        if len(current_code) > 1 and move_type < 0.4:
            # Remove a random codeword
            idx_to_remove = random.randint(0, len(current_code) - 1)
            c = current_code[idx_to_remove]
            
            # Check if removal causes uncovered words
            # If coverage_count[ball_mask[c]] > 1 everywhere, safe.
            # If any becomes 0, we have a penalty.
            
            # Tentative update
            ball_c = ball_mask[c]
            new_cov_count = current_coverage_count.copy()
            new_cov_count[ball_c] -= 1
            
            new_cost = get_cost(current_code, new_cov_count) # Note: len changes by -1
            # Actually, get_cost uses len(code). Let's pass tentative len.
            # Let's refine get_cost to take explicit len
            def calc_cost(len_code, cov_count):
                unc = np.sum(cov_count == 0)
                return len_code + lambda_uncovered * unc
            
            new_cost_val = calc_cost(len(current_code) - 1, new_cov_count)
            
            delta = new_cost_val - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_code.pop(idx_to_remove)
                current_coverage_count = new_cov_count
                current_cost = new_cost_val
                
                # Check if better valid solution
                if new_cov_count.min() > 0 and len(current_code) < best_cost:
                    best_cost = len(current_code)
                    best_code = current_code.copy()
                    
        elif move_type < 0.7:
            # Swap: Remove one, Add one
            if len(current_code) == 0:
                continue
            idx_to_remove = random.randint(0, len(current_code) - 1)
            c_old = current_code[idx_to_remove]
            
            # Pick a random new codeword
            c_new = random.randint(0, N - 1)
            
            # If c_new is already in code, it's just a no-op or duplicate (not allowed)
            if c_new in current_code:
                continue
                
            # Tentative update
            ball_old = ball_mask[c_old]
            ball_new = ball_mask[c_new]
            
            new_cov_count = current_coverage_count.copy()
            new_cov_count[ball_old] -= 1
            new_cov_count[ball_new] += 1
            
            new_cost_val = calc_cost(len(current_code), new_cov_count) # Size unchanged
            
            delta = new_cost_val - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_code[idx_to_remove] = c_new
                current_coverage_count = new_cov_count
                current_cost = new_cost_val
                
                if new_cov_count.min() > 0 and len(current_code) < best_cost:
                    best_cost = len(current_code)
                    best_code = current_code.copy()
                    
        else:
            # Add a random codeword
            c_new = random.randint(0, N - 1)
            if c_new in current_code:
                continue
                
            ball_new = ball_mask[c_new]
            new_cov_count = current_coverage_count.copy()
            new_cov_count[ball_new] += 1
            
            new_cost_val = calc_cost(len(current_code) + 1, new_cov_count)
            
            delta = new_cost_val - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_code.append(c_new)
                current_coverage_count = new_cov_count
                current_cost = new_cost_val
                
                # Adding doesn't improve size, but might help future removals
                # We don't update best_code here because size increased.
                
        T *= cooling_rate
        
    # Final Reduction Pass on Best Code
    # Try to remove redundant words from best_code
    best_code = list(best_code)
    # Rebuild coverage for best_code
    final_cov = np.zeros(N, dtype=np.int32)
    for c in best_code:
        final_cov[ball_mask[c]] += 1
        
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(len(best_code) - 1, -1, -1):
            if len(best_code) <= 1:
                break
            c = best_code[i]
            # Check if removal is safe
            # If final_cov[ball_mask[c]] > 1 for all w in ball, then safe.
            # i.e. min(final_cov[ball_mask[c]]) > 1
            if np.min(final_cov[ball_mask[c]]) > 1:
                best_code.pop(i)
                final_cov[ball_mask[c]] -= 1
                changed = True
                break # Restart loop as indices changed

    # Final validity check
    if np.any(final_cov == 0):
        # Fallback to initial greedy
        return [list(all_words[i]) for i in code_set]
        
    return [list(all_words[i]) for i in best_code]
# EVOLVE-BLOCK-END
