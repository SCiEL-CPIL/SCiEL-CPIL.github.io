import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit (evaluator kills at 150s, but keep margin)
    
    N = 1000
    best_set = []
    best_size = 0
    
    def run_greedy_smart(random_tie_break=True):
        """
        Greedy with heuristic: pick valid candidate that minimizes 'tightness'.
        Uses NumPy for vectorized validity and score checks.
        """
        chosen = []
        # sum_counts[s] is the number of pairs (a,b) with a<=b from chosen such that a+b=s
        sum_counts = np.zeros(2 * N + 1, dtype=np.int32)
        is_chosen = np.zeros(N + 1, dtype=bool)
        
        # List of remaining candidates
        remaining = list(range(1, N + 1))
        
        while remaining:
            if time.time() > deadline - 5:
                break
                
            # Convert remaining to numpy array for vectorization
            cands = np.array(remaining, dtype=np.int32)
            
            if not chosen:
                # First element can be anything, but to keep consistency with smart greedy
                # we might just pick the one with min score (all 0) or random.
                # In original, score is 0 for all. 
                # Let's just pick the first one or random.
                if random_tie_break:
                    x = random.choice(remaining)
                else:
                    x = remaining[0]
                
                remaining.remove(x)
                chosen.append(x)
                is_chosen[x] = True
                sum_counts[2 * x] += 1
                continue
                
            chosen_arr = np.array(chosen, dtype=np.int32)
            
            # Calculate sums: (num_cands, num_chosen)
            # sums[i, j] = cands[i] + chosen_arr[j]
            # We need to check if any sum has count >= 2.
            # Also check 2*x.
            
            # Optimization: Only consider candidates that don't immediately violate 2*x
            # But vectorizing the full check is easier.
            
            sums = cands[:, None] + chosen_arr[None, :]
            
            # Get counts for these sums
            # sum_counts is 1D, sums is 2D. 
            # sum_counts[sums] works via broadcasting if sum_counts is 1D? 
            # Yes, sum_counts[sums] will have shape (num_cands, num_chosen)
            counts_at_sums = sum_counts[sums]
            
            # Validity: No sum should have count >= 2
            # If any count in row i is >= 2, candidate i is invalid.
            max_count_per_cand = counts_at_sums.max(axis=1)
            
            # Also check 2*x
            double_sums = 2 * cands
            double_counts = sum_counts[double_sums]
            
            valid_mask = (max_count_per_cand < 2) & (double_counts < 2)
            
            if not np.any(valid_mask):
                break
                
            # Calculate score for valid candidates
            # Score is number of sums that would become count=2 (i.e., currently count=1)
            # Plus 1 if 2*x is currently count=1.
            
            # For valid candidates, counts_at_sums are all < 2.
            # So count=1 means it will become 2.
            # count=0 means it will become 1.
            
            # We only care about valid candidates.
            # Initialize score array with -1 or large number for invalid
            scores = np.full(len(cands), np.inf, dtype=np.int32)
            
            if np.any(valid_mask):
                # Sum of (counts_at_sums == 1) for valid rows
                # We can compute this for all rows and mask later
                num_ones = (counts_at_sums == 1).sum(axis=1)
                num_double_ones = (double_counts == 1).astype(np.int32)
                
                total_scores = num_ones + num_double_ones
                
                # Apply mask
                scores[valid_mask] = total_scores[valid_mask]
                
                # Find min score
                min_score = np.min(scores[valid_mask])
                
                # Get candidates with min score
                # valid_indices = np.where(valid_mask)[0]
                # min_indices_in_valid = np.where(scores[valid_mask] == min_score)[0]
                # This is tricky with masking.
                
                # Alternative: Set scores of invalid to inf, then argmin
                min_score_val = np.min(scores)
                best_cand_indices = np.where(scores == min_score_val)[0]
                
                if len(best_cand_indices) == 0:
                    break
                    
                if random_tie_break and len(best_cand_indices) > 1:
                    idx = random.choice(best_cand_indices)
                else:
                    idx = best_cand_indices[0]
                    
                x = cands[idx]
                
            else:
                break
                
            # Update state
            remaining.remove(x)
            chosen.append(x)
            is_chosen[x] = True
            
            # Update sum_counts
            # Add 1 to sum_counts[x + y] for all y in chosen (including x? No, x is new)
            # The loop `for y in chosen` in original included x?
            # Original:
            # for y in chosen:
            #     if y == x: continue
            #     s = x + y
            #     counts[s] += 1
            # counts[2 * x] += 1
            
            # So we add 1 to x + y for all y in old_chosen.
            # And 1 to 2*x.
            
            if chosen_arr is not None and len(chosen_arr) > 0:
                new_sums = x + chosen_arr
                sum_counts[new_sums] += 1
                
            sum_counts[2 * x] += 1
            
        return chosen

    def run_greedy_random():
        """
        Greedy with random order.
        """
        chosen = []
        sum_counts = np.zeros(2 * N + 1, dtype=np.int32)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 5:
                break
                
            # Check validity
            # We can use numpy here too, but simple loop might be fast enough for random greedy
            # since we don't evaluate all candidates.
            # But let's use numpy for consistency and speed.
            if chosen:
                chosen_arr = np.array(chosen, dtype=np.int32)
                sums = x + chosen_arr
                counts_at_sums = sum_counts[sums]
                if np.any(counts_at_sums >= 2):
                    continue
                if sum_counts[2 * x] >= 2:
                    continue
            
            chosen.append(x)
            if len(chosen) > 1:
                # Update counts
                # x + all previous
                prev_arr = np.array(chosen[:-1], dtype=np.int32)
                new_sums = x + prev_arr
                sum_counts[new_sums] += 1
            sum_counts[2 * x] += 1
                
        return chosen

    # 1. Try Smart Greedy with random tie-breaking
    # Since it's faster now, we can do more iterations
    num_smart_restarts = 50
    
    for i in range(num_smart_restarts):
        if time.time() > deadline - 10:
            break
        s = run_greedy_smart(random_tie_break=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 2. Try Random Greedy Restarts
    # Use remaining time
    attempts = 0
    while time.time() < deadline - 2:
        s = run_greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        attempts += 1
        
        # Check time every 50 attempts to avoid overhead
        if attempts % 50 == 0 and time.time() > deadline - 5:
            break

    return best_set
