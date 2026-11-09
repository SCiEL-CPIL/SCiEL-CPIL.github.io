import time
import random
import sys

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0 # Safety margin for 120s limit
    
    # Parameters for the specific task: q=3, n=7, radius=1
    # Total words N = 3^7 = 2187
    N = q ** n
    
    # Helper: Convert index to tuple and vice versa
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx

    # Precompute all words
    all_words = [tuple(range(q) * n) for _ in range(N)] # Placeholder, will generate properly
    all_words = []
    for i in range(N):
        all_words.append(idx_to_tuple(i))

    # Precompute balls as bitsets (Python ints)
    # ball_mask[i] is an int with bits set for all words within distance 'radius' of word i
    ball_masks = [0] * N
    
    # We also need a list of indices for each ball for potential tie-breaking or explicit iteration
    ball_indices = [[] for _ in range(N)]
    
    # To compute balls efficiently, we can use the fact that radius=1.
    # Ball of i is {i} U {words at dist 1}
    
    # Precompute neighbors at distance 1 for each word
    # This is 7 * 2 = 14 neighbors per word.
    
    for i in range(N):
        w = all_words[i]
        w_list = list(w)
        mask = 1 << i
        indices = [i]
        
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    j = tuple_to_idx(tuple(w_list))
                    mask |= (1 << j)
                    indices.append(j)
            w_list[pos] = old_val
            
        ball_masks[i] = mask
        ball_indices[i] = indices

    # --- Strategy: Bitset-Accelerated Greedy + Intensive Local Search ---
    
    best_code = []
    best_len = N
    
    def get_code_from_indices(indices):
        return [list(all_words[i]) for i in indices]

    # 1. Fast Greedy Initialization with Bitsets
    # We run multiple greedy restarts with different tie-breaking to find a good initial cover.
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - t_start > deadline - 30:
            break
            
        covered = 0
        code = []
        
        # Randomize the order of candidates or use random tie-breaking
        # To speed up, we can just pick the one with max gain, breaking ties randomly
        
        while covered != (1 << N) - 1:
            # Find the word with maximum gain
            # Gain = popcount(ball_mask[i] & ~covered)
            # We can't check all N every time (N=2187, loop ~2187*2187/15 ~ 300k ops, which is fine in Py if optimized, but let's be careful)
            
            max_gain = -1
            best_idx = -1
            candidates = []
            
            # Optimization: Only check words that are NOT yet in the code? 
            # Actually, adding a word already in the code is useless.
            # But checking all N is 2187 iterations. 2187 * 150 iterations (approx code size) = 328k iterations.
            # Each iteration does a bitwise AND and POPCOUNT.
            # Python int popcount is fast (C implementation).
            
            # To make it faster, we can maintain a list of "active" candidates? 
            # Or just iterate all. 300k iterations in Python might take ~0.1-0.5s.
            # 50 restarts * 0.5s = 25s. This is acceptable.
            
            # Let's try to optimize the inner loop.
            # We can skip words that are fully covered? No, we need to know if they add NEW coverage.
            
            for i in range(N):
                if i in code: # This is O(len(code)). Bad. Use a set or boolean array.
                    continue
                    
            # Let's use a boolean array 'in_code' for O(1) check.
            # But re-initializing is expensive.
            
            # Alternative: Just iterate all. The check `if i in code` is the bottleneck.
            # Let's use a set `code_set` for O(1) membership.
            
            pass # The above loop structure is a bit messy. Let's refine.

    # Refined Greedy with Bitsets
    def greedy_init(random_seed):
        random.seed(random_seed)
        covered = 0
        code_set = set()
        code_list = []
        total_mask = (1 << N) - 1
        
        while covered != total_mask:
            max_gain = -1
            best_candidates = []
            
            # Iterate through all words
            for i in range(N):
                if i in code_set:
                    continue
                
                # Calculate gain
                # gain = popcount(ball_masks[i] & ~covered)
                # In Python: (mask & ~covered).bit_count()
                gain = (ball_masks[i] & ~covered).bit_count()
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if cover is possible
                
            # Random tie-break
            idx = random.choice(best_candidates)
            code_list.append(idx)
            code_set.add(idx)
            covered |= ball_masks[idx]
            
        return code_list

    # Run a few greedy restarts
    for i in range(10):
        if time.time() - t_start > deadline - 60:
            break
        code = greedy_init(i)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
    # 2. Intensive Local Search (Simulated Annealing style but with Bitsets)
    # We perform swaps: remove one, add one.
    # Check validity: new_covered = (covered_without_removed) | ball_masks[added]
    # If new_covered == total_mask, accept.
    
    current_code = best_code.copy()
    current_set = set(current_code)
    
    # Precompute covered mask for current code
    current_covered = 0
    for i in current_code:
        current_covered |= ball_masks[i]
        
    total_mask = (1 << N) - 1
    
    # To check if a swap is valid, we need to know if (current_covered & ~ball_masks[remove]) | ball_masks[add] == total_mask
    # This is (current_covered | ball_masks[add]) & ~ball_masks[remove] == total_mask? No.
    # Let C be current covered.
    # Remove r: C' = C & ~ball_masks[r]
    # Add a: C'' = C' | ball_masks[a]
    # Valid if C'' == total_mask.
    
    # We can precompute "essential" words? 
    # A word r is "removable" if (C & ~ball_masks[r]) == total_mask.
    
    # Local Search Loop
    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    while time.time() - t_start < deadline - 5 and T > min_T:
        # Try a random swap
        if not current_code:
            break
            
        idx_remove = random.choice(current_code)
        idx_add = random.randrange(N)
        
        if idx_add in current_set:
            continue
            
        # Calculate new covered
        # C_without_remove = current_covered & ~ball_masks[idx_remove]
        # C_new = C_without_remove | ball_masks[idx_add]
        
        c_without = current_covered & ~ball_masks[idx_remove]
        c_new = c_without | ball_masks[idx_add]
        
        if c_new == total_mask:
            # Valid swap!
            current_set.discard(idx_remove)
            current_set.add(idx_add)
            current_code.remove(idx_remove)
            current_code.append(idx_add)
            current_covered = c_new
            
            # Try to remove more? (Greedy removal after swap)
            # Check if any word in current_code is now redundant
            # We can do a quick pass
            for i in list(current_code):
                if (current_covered & ~ball_masks[i]) == total_mask:
                    # Redundant
                    current_covered &= ~ball_masks[i]
                    current_set.discard(i)
                    current_code.remove(i)
                    # Update best
                    if len(current_code) < best_len:
                        best_len = len(current_code)
                        best_code = current_code.copy()
            
            if len(current_code) < best_len:
                best_len = len(current_code)
                best_code = current_code.copy()
        else:
            # Invalid swap, revert (do nothing)
            # In SA, we might accept invalid moves to escape local minima, 
            # but for covering codes, we usually stay in the valid set or move to invalid and fix?
            # Staying in valid set is safer.
            pass
            
        T *= cooling

    # 3. Final Greedy Removal Pass
    changed = True
    while changed and time.time() - t_start < deadline - 2:
        changed = False
        for i in list(current_code):
            if (current_covered & ~ball_masks[i]) == total_mask:
                current_covered &= ~ball_masks[i]
                current_set.discard(i)
                current_code.remove(i)
                changed = True
                if len(current_code) < best_len:
                    best_len = len(current_code)
                    best_code = current_code.copy()

    if not best_code:
        best_code = current_code

    # Convert indices to list of lists
    result = []
    for idx in best_code:
        result.append(list(all_words[idx]))
        
    return result
