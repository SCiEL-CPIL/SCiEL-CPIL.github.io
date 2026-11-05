import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # 1. Generate standard recursive superpermutation
    # This is the "canonical" superpermutation
    # S_1 = "1"
    # S_m = S_{m-1} + m + S_{m-1} (with overlaps)
    # Actually, the standard construction is:
    # S_1 = 1
    # S_2 = 1 2 1
    # S_3 = 1 2 1 3 1 2 1
    # S_4 = 1 2 1 3 1 2 1 4 1 2 1 3 1 2 1 ...
    # The length grows as: L_1=1, L_2=3, L_3=9, L_4=33, L_5=153, L_6=873, L_7=6513
    
    word = [1]
    for m in range(2, n + 1):
        # Extract all unique (m-1)-permutations from the current word
        # They appear in order.
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Overlap with current end of new_word
            # Find max overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
        
    S = list(word)
    L = len(S)
    
    # Initialize counts
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = L
    best_counts = counts.copy() # Not strictly needed if we restore S, but good for reference
    
    def try_remove(S, counts, i):
        """
        Attempts to remove S[i]. Returns (success, new_S, new_counts)
        If successful, S and counts are modified in place? 
        Better to return new structures to allow rollback easily.
        """
        n_local = n
        # Old windows affected: indices j such that window [j, j+n) contains i
        # j <= i < j+n  =>  i-n+1 <= j <= i
        start_old = max(0, i - n_local + 1)
        end_old = min(i, len(S) - n_local)
        
        old_windows = []
        for j in range(start_old, end_old + 1):
            old_windows.append(tuple(S[j:j+n_local]))
            
        # Check validity: 
        # Any old window with count 1 must be present in the new windows
        # New windows are formed by S[:i] + S[i+1:]
        # The new windows that change are those straddling i.
        # In the new string S', the index i is removed.
        # Windows in S' that correspond to old windows straddling i:
        # The new string length is L-1.
        # The windows affected are at indices k in S' where the window covers the "gap".
        # In S', the gap is at i.
        # Windows k in S' such that k < i <= k+n-1?
        # Actually, it's easier to just construct the new windows that appear.
        # The new windows are:
        # For k in range(start_new, end_new + 1):
        #   window = S'[k:k+n]
        
        # Let's just check if any old window p has counts[p] == 1 and p is NOT in the set of new windows.
        
        # Construct set of new windows that are "new" or "shifted"
        # Actually, only windows that straddle the removal point are "new" in terms of content.
        # Windows entirely before i are unchanged.
        # Windows entirely after i are unchanged (content wise, just shifted index).
        # So we only need to check the windows that straddle i in the new string.
        
        # In S, the removal is at i.
        # In S', the characters S[i+1] ... S[i+n-1] are now adjacent to S[i-1].
        # The new windows are centered around the junction.
        # Specifically, the windows in S' at indices k such that the window includes S'[i-1] and S'[i] (which is S[i+1] in S).
        # Indices in S':
        # The junction is between S'[i-1] and S'[i].
        # Windows covering this junction:
        # k such that k <= i-1 and k+n-1 >= i.
        # i-n+1 <= k <= i-1.
        
        start_new = max(0, i - n_local + 1)
        end_new = min(i - 1, len(S) - 1 - n_local)
        
        new_windows_set = set()
        for k in range(start_new, end_new + 1):
            # Construct window from S
            # S'[k:k+n]
            # If idx < i: S[idx]
            # If idx >= i: S[idx+1]
            w = []
            for idx in range(k, k + n_local):
                if idx < i:
                    w.append(S[idx])
                else:
                    w.append(S[idx + 1])
            new_windows_set.add(tuple(w))
            
        # Check old windows
        for p in old_windows:
            if counts.get(p, 0) == 1:
                if p not in new_windows_set:
                    return False, None, None
                    
        # If valid, perform removal
        # Update counts
        # Decrement old
        for p in old_windows:
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
                
        # Increment new
        # Note: new_windows_set might have duplicates if the window repeats?
        # No, k is unique, so the window position is unique.
        # But the tuple value might be the same.
        # We need to increment the count for each new window instance.
        for k in range(start_new, end_new + 1):
            w = []
            for idx in range(k, k + n_local):
                if idx < i:
                    w.append(S[idx])
                else:
                    w.append(S[idx + 1])
            p_new = tuple(w)
            counts[p_new] = counts.get(p_new, 0) + 1
            
        # Remove from S
        S.pop(i)
        
        return True, S, counts

    # Main Search Loop
    while time.time() < deadline:
        # 1. Greedy Removal Pass (Random Order)
        indices = list(range(len(S)))
        random.shuffle(indices)
        made_change = False
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(S):
                continue
            # Ensure i is still valid index (if previous removals shifted things? 
            # We are iterating over a shuffled list of original indices? 
            # If we remove an element, indices change.
            # It is safer to restart the pass or use a set of valid indices.
            # Given the cost of removal is low, restarting is often better for finding more removals.
            # But restarting is O(L^2) worst case.
            # Let's try a single pass with careful index management.
            # Actually, if we remove index i, all indices > i are shifted left by 1.
            # Our shuffled list is static.
            # If we process i, and it's valid, we remove it.
            # Next index j in list:
            # If j < i, it's still valid.
            # If j > i, it now points to S[j-1].
            # This is messy.
            
            # Strategy: Just try to remove. If index is out of bounds, skip.
            # This is a heuristic.
            
            success, new_S, new_counts = try_remove(S, counts, i)
            if success:
                S = new_S
                counts = new_counts
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                # Break to restart pass? 
                # If we continue, we might miss removals that became valid due to the shift.
                # But we also might remove things that are no longer "good".
                # Restarting is safer and often finds a "minimal" superpermutation faster.
                break
        
        if made_change:
            continue
            
        # 2. If no single removal, try Pair Removal
        made_change = False
        # Generate random pairs
        # L is ~6000. Pairs ~6000.
        # Try a random subset of pairs to save time?
        # Or just iterate all pairs in random order.
        pair_indices = list(range(len(S) - 1))
        random.shuffle(pair_indices)
        
        for i in pair_indices:
            if time.time() > deadline:
                break
            if i >= len(S) - 1:
                continue
            
            # Try removing i and i+1
            # This is equivalent to removing i, then removing i (which is now i+1 in the old indexing, but i in new).
            # Let's implement a specific pair removal check.
            
            # Old windows affected by removing i and i+1:
            # Union of windows affected by removing i and removing i+1.
            # Start: max(0, i-n+1, i+1-n+1) = max(0, i-n+1)
            # End: min(i, i+1, L-n) = min(i+1, L-n)
            
            start_old = max(0, i - n + 1)
            end_old = min(i + 1, len(S) - n)
            
            old_windows = []
            for j in range(start_old, end_old + 1):
                old_windows.append(tuple(S[j:j+n]))
                
            # New windows:
            # Removing i and i+1 joins S[i-1] and S[i+2].
            # Junction is between new index i-1 and i.
            # Windows straddling this junction in new string.
            # New length L-2.
            # Start new: max(0, i - n + 1)
            # End new: min(i - 1, L - 2 - n)
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n)
            
            new_windows_set = set()
            for k in range(start_new, end_new + 1):
                w = []
                for idx in range(k, k + n):
                    # Map new idx to old S
                    # if idx < i-1: S[idx]
                    # if idx == i-1: S[i-1]
                    # if idx >= i: S[idx+2]
                    if idx < i - 1:
                        w.append(S[idx])
                    elif idx == i - 1:
                        w.append(S[i - 1])
                    else:
                        w.append(S[idx + 2])
                new_windows_set.add(tuple(w))
                
            # Check validity
            valid = True
            for p in old_windows:
                if counts.get(p, 0) == 1:
                    if p not in new_windows_set:
                        valid = False
                        break
                        
            if valid:
                # Perform removal
                # Decrement old
                for p in old_windows:
                    counts[p] -= 1
                    if counts[p] == 0:
                        del counts[p]
                        
                # Increment new
                for k in range(start_new, end_new + 1):
                    w = []
                    for idx in range(k, k + n):
                        if idx < i - 1:
                            w.append(S[idx])
                        elif idx == i - 1:
                            w.append(S[i - 1])
                        else:
                            w.append(S[idx + 2])
                    p_new = tuple(w)
                    counts[p_new] = counts.get(p_new, 0) + 1
                    
                S.pop(i)
                S.pop(i) # Remove the next one (which is now at i)
                
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                break
                
        if made_change:
            continue
            
        # 3. If stuck on singles and pairs, try Perturbation
        # Random Swap
        if len(S) > n:
            # Pick two random indices
            i = random.randint(0, len(S) - 1)
            j = random.randint(0, len(S) - 1)
            if i != j:
                # Check if swap is valid
                # Swap S[i] and S[j].
                # This affects windows around i and around j.
                # If |i-j| is small, they overlap.
                # For simplicity, only try adjacent swaps or small distance swaps.
                # Let's try adjacent swap i, i+1.
                i = random.randint(0, len(S) - 2)
                j = i + 1
                
                # Check validity of swap
                # Old windows: around i and j.
                # New windows: around i and j.
                
                # This is getting complex. 
                # Alternative: Just break and return best.
                # The greedy removal usually gets us close.
                # 6513 -> ~6000 with greedy.
                # 5905 is the target.
                # Maybe we can do a "Random Deletion and Repair"? No, too slow.
                
                # Let's try a simple adjacent swap check.
                # If valid, perform it.
                
                # Windows affected by swapping i, i+1:
                # Old: j in [i-n+1, i+1]
                # New: j in [i-n+1, i+1]
                
                old_w = []
                for k in range(max(0, i-n+1), min(i+1, len(S)-n) + 1):
                    old_w.append(tuple(S[k:k+n]))
                    
                # Create temp S
                S_temp = S[:]
                S_temp[i], S_temp[i+1] = S_temp[i+1], S_temp[i]
                
                new_w = []
                for k in range(max(0, i-n+1), min(i+1, len(S)-n) + 1):
                    new_w.append(tuple(S_temp[k:k+n]))
                    
                # Check if any old window with count 1 is not in new_w
                # Note: new_w might have duplicates.
                new_w_set = set(new_w)
                
                valid_swap = True
                for p in old_w:
                    if counts.get(p, 0) == 1:
                        if p not in new_w_set:
                            valid_swap = False
                            break
                            
                if valid_swap:
                    # Perform swap
                    # Update counts
                    for p in old_w:
                        counts[p] -= 1
                        if counts[p] == 0:
                            del counts[p]
                    for p in new_w:
                        counts[p] = counts.get(p, 0) + 1
                        
                    S[i], S[i+1] = S[i+1], S[i]
                    made_change = True
                    # Don't update best_len yet, as length is same.
                    # But this might enable removals.
                    
        if not made_change:
            # Stuck completely.
            # Try a few more random shuffles of the removal order?
            # Or just break.
            # Given the time limit, breaking is safer.
            # We can try to restart the whole process with a different random seed?
            # No, we are in the loop.
            # Let's try to do a "Deep" removal: remove 3 chars?
            # Too complex.
            break
            
    return "".join(map(str, best_S))
