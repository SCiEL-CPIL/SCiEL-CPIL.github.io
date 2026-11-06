import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # 1. Precompute APs
    # APs are (start, step)
    # We need indices for each AP
    # And for each index, which APs contain it
    
    # To save memory and time, we can just generate the lists.
    aps_indices = [] # List of lists of indices
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1: # Task says "including one", but sum of 1 is 1, max abs is 1. 
                               # If we allow length 1, max abs is at least 1.
                               # The task says "including one". 
                               # If max abs is 1, we can't do better. 
                               # But we can do better than 1? 
                               # Sum of 1 term is +/- 1. So max abs >= 1.
                               # If we find a config with max abs 1, it's optimal.
                               # But usually it's higher.
                               # Including length 1 APs means the max is at least 1.
                               # The current program ignores length 1 APs (len > 1).
                               # If the evaluator includes length 1, the score is max(1, max_len_ge_2).
                               # Since 1 is small, minimizing max_len_ge_2 is still the goal.
                               # So we can ignore length 1 APs in our search, as they don't constrain the "max" if max > 1.
                               # If we achieve max=1, we are done.
                               # So ignoring len=1 is safe for optimization, as long as we check final result.
                               pass
            
            if len(terms) > 1:
                ap_id = len(aps_indices)
                aps_indices.append(terms)
                for idx in terms:
                    aps_containing[idx].append(ap_id)
                
    num_aps = len(aps_indices)
    
    # Histogram size: sums can be -256 to 256.
    # Offset 256. Size 513.
    HIST_SIZE = 2 * N + 1
    OFFSET = N
    
    def init_histogram(ap_sums):
        hist = np.zeros(HIST_SIZE, dtype=np.int32)
        for s in ap_sums:
            hist[s + OFFSET] += 1
        return hist

    def get_max_from_hist(hist):
        # Scan from edges
        # max possible is N
        for val in range(N, -1, -1):
            if hist[val + OFFSET] > 0 or hist[-val + OFFSET] > 0:
                return val
        return 0

    # Greedy Init
    def greedy_init():
        signs = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        hist = np.zeros(HIST_SIZE, dtype=np.int32)
        current_max = 0
        
        # Initialize hist with all zeros? No, ap_sums are 0.
        # All APs have sum 0.
        hist[0 + OFFSET] = num_aps
        current_max = 0 # Max abs of 0 is 0? No, we are building it.
        
        # Actually, let's just place signs one by one.
        # Initially all sums are 0.
        
        for i in range(N):
            best_sign = 1
            best_max = N + 1
            best_sums = None
            best_hist = None
            
            for s in [1, -1]:
                # Calculate new sums and hist
                new_sums = ap_sums.copy()
                new_hist = hist.copy()
                
                # Update sums for APs containing i
                for ap_idx in aps_containing[i]:
                    old_sum = new_sums[ap_idx]
                    new_sum = old_sum + s
                    new_sums[ap_idx] = new_sum
                    new_hist[old_sum + OFFSET] -= 1
                    new_hist[new_sum + OFFSET] += 1
                
                # Find max
                # Optimization: if we know current_max, we can check if it changed.
                # But for init, just compute.
                m = get_max_from_hist(new_hist)
                
                if m < best_max:
                    best_max = m
                    best_sign = s
                    best_sums = new_sums
                    best_hist = new_hist
            
            signs[i] = best_sign
            ap_sums = best_sums
            hist = best_hist
            current_max = best_max
            
        return signs, ap_sums, hist, current_max

    def local_search(signs, ap_sums, hist, current_max, time_limit):
        # 2-flip local search
        while time.time() < time_limit:
            # Pick 2 random indices
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            
            if i == j:
                continue
                
            # Get affected APs
            affected_aps = set(aps_containing[i]) | set(aps_containing[j])
            
            # Store old values
            old_sums = {}
            
            # Apply flip
            # Delta for i: -2 * signs[i]
            # Delta for j: -2 * signs[j]
            di = -2 * signs[i]
            dj = -2 * signs[j]
            
            for ap_idx in affected_aps:
                old_sum = ap_sums[ap_idx]
                new_sum = old_sum
                # Check if i is in ap
                # We can precompute membership? 
                # Or just check if ap_idx is in aps_containing[i]
                # This is O(1) with a set, but we have lists.
                # Better: Precompute a boolean matrix or just check.
                # Since num_aps is ~60k, and affected is ~3k, 
                # checking membership in list is O(len). 
                # Let's precompute a sparse structure or just use the list of APs for i and j.
                
                # Actually, we can just iterate through aps_containing[i] and aps_containing[j]
                # and handle duplicates.
                
                pass
            
            # This "set" approach is slow in Python.
            # Let's stick to the list iteration and handle duplicates carefully.
            
            # Re-implementing the move logic for speed:
            
            # 1. Identify unique affected APs
            # We can just iterate through aps_containing[i] and aps_containing[j]
            # and use a dictionary to store old sums.
            
            old_sums_dict = {}
            
            # Process i
            for ap_idx in aps_containing[i]:
                if ap_idx not in old_sums_dict:
                    old_sums_dict[ap_idx] = ap_sums[ap_idx]
                    # Apply delta i
                    ap_sums[ap_idx] += di
                    # Update hist
                    hist[old_sums_dict[ap_idx] + OFFSET] -= 1
                    hist[ap_sums[ap_idx] + OFFSET] += 1
            
            # Process j
            for ap_idx in aps_containing[j]:
                if ap_idx not in old_sums_dict:
                    old_sums_dict[ap_idx] = ap_sums[ap_idx]
                    # Apply delta j
                    ap_sums[ap_idx] += dj
                    hist[old_sums_dict[ap_idx] + OFFSET] -= 1
                    hist[ap_sums[ap_idx] + OFFSET] += 1
                else:
                    # Already processed i, now apply j
                    old_sum_before_j = ap_sums[ap_idx]
                    ap_sums[ap_idx] += dj
                    hist[old_sum_before_j + OFFSET] -= 1
                    hist[ap_sums[ap_idx] + OFFSET] += 1
            
            # Find new max
            # Optimization: 
            # If we only decreased sums, max might decrease.
            # If we increased any sum to > current_max, max increased.
            
            # Simple: recompute max from hist.
            new_max = get_max_from_hist(hist)
            
            accepted = False
            if new_max < current_max:
                accepted = True
            elif new_max == current_max:
                # Random acceptance to escape plateaus?
                if random.random() < 0.1:
                    accepted = True
            
            if not accepted:
                # Revert
                for ap_idx, old_sum in old_sums_dict.items():
                    # Current sum is in ap_sums[ap_idx]
                    cur_sum = ap_sums[ap_idx]
                    ap_sums[ap_idx] = old_sum
                    hist[cur_sum + OFFSET] -= 1
                    hist[old_sum + OFFSET] += 1
                # Revert signs
                signs[i] = -signs[i] # Wait, we didn't change signs yet!
                # We should only change signs if accepted.
                # So we need to revert the "virtual" sign change?
                # No, we haven't changed signs[i] or signs[j] in the array yet.
                # We only changed ap_sums and hist.
                # So we just revert ap_sums and hist.
                
                # Wait, the delta calculation used signs[i].
                # If we revert, signs[i] is still the old value.
                # So next time we use signs[i], it's correct.
                
                # So we just revert ap_sums and hist.
                pass
            else:
                # Accept
                signs[i] = -signs[i]
                signs[j] = -signs[j]
                current_max = new_max

        return signs, ap_sums, hist, current_max

    best_val = N + 1
    best_signs = None
    
    while time.time() < deadline:
        # Greedy init
        signs, ap_sums, hist, current_max = greedy_init()
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
            
        if best_val <= 2:
            break
            
        # Local search
        time_left = deadline - 2.0
        signs, ap_sums, hist, current_max = local_search(signs, ap_sums, hist, current_max, time_left)
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
            
    return best_signs.tolist()
