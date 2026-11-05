import sys
from collections import Counter

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Step 2: Local Search to minimize length
    # We represent the word as a list of ints for easier manipulation
    # We need to track which permutations are present.
    # There are n! permutations.
    
    # Generate all permutations
    import itertools
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Count occurrences in the current word
    # A window is valid if it is a permutation.
    # We maintain a count array of size n!
    
    def get_initial_counts(w):
        counts = [0] * (len(all_perms))
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if window in perm_to_idx:
                counts[perm_to_idx[window]] += 1
        return counts

    counts = get_initial_counts(word)
    
    # Check if current word is valid (all counts > 0)
    def is_valid(counts):
        return all(c > 0 for c in counts)
        
    if not is_valid(counts):
        # Should not happen with standard construction
        return "".join(map(str, word))

    # Heuristic Search
    # Time limit management
    import time
    start_time = time.time()
    time_limit = 100 # Leave 20s for safety
    
    # We will try to remove characters.
    # To make it efficient, we only update counts for affected windows.
    # Removing char at index i affects windows starting at i-n+1 to i.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try all positions for deletion
        for i in range(len(word) - 1, -1, -1):
            if (time.time() - start_time) > time_limit:
                break
            
            # Calculate new counts if we remove word[i]
            # Windows affected: start indices [i-n+1, i]
            # Old windows:
            old_windows = []
            for k in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                w = tuple(word[k:k + n])
                if w in perm_to_idx:
                    old_windows.append(perm_to_idx[w])
            
            # New windows after removal:
            # The string becomes word[:i] + word[i+1:]
            # New windows start at indices in the new string.
            # A window in the new string corresponds to a window in the old string 
            # that did NOT include index i, OR is formed by merging parts around i.
            
            # This manual update is error-prone. 
            # Alternative: Just recompute counts for the modified string?
            # O(N * n) to extract windows, O(1) to check perm.
            # N ~ 6000. 6000 * 7 = 42000 ops per check.
            # 42000 * 6000 positions = 250 million ops. Too slow in Python.
            
            # We need the incremental update.
            # Let's implement the incremental update carefully.
            
            # 1. Decrement counts for windows in old string that contain index i.
            #    These are windows starting at k where i-n+1 <= k <= i.
            #    And k must be a valid start index (0 <= k <= len(word)-n).
            
            temp_counts = counts[:] # Copy is expensive (size 5040). 
            # 5040 ints copy is fast (~microseconds).
            
            for k in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                w = tuple(word[k:k + n])
                if w in perm_to_idx:
                    temp_counts[perm_to_idx[w]] -= 1
            
            # 2. Increment counts for windows in new string that contain the "gap" at i.
            #    The new string is w_new = word[:i] + word[i+1:]
            #    The windows in w_new that are "new" (i.e., span the deletion point) 
            #    are those starting at k in w_new such that the window covers the 
            #    junction between word[i-1] and word[i+1].
            #    In w_new, the junction is at index i-1 (0-indexed in w_new, 
            #    corresponding to old index i-1).
            #    The window starts at k_new. It covers indices k_new ... k_new+n-1.
            #    It spans the junction if k_new <= i-1 < k_new+n.
            #    So k_new can be i-1, i-2, ..., i-n.
            #    Also k_new must be >= 0 and k_new <= len(w_new)-n.
            
            len_new = len(word) - 1
            for k_new in range(max(0, i - n), min(i, len_new - n + 1)):
                # Construct the window from w_new
                # Indices in w_new:
                # w_new[j] = word[j] if j < i else word[j+1]
                # Window is w_new[k_new : k_new+n]
                # This corresponds to:
                # If k_new + n <= i: word[k_new : k_new+n] (Unchanged, already counted? 
                #    No, we only decremented old windows containing i.
                #    If the window is entirely before i, it didn't contain i, so it wasn't decremented.
                #    It is still present in the new string. So we don't need to increment it again.
                #    Wait. The set of windows in the new string is:
                #    1. Windows entirely before i (start < i-n+1). Unchanged.
                #    2. Windows entirely after i (start > i). Unchanged (indices shifted, but content same).
                #    3. Windows spanning i. These are the ones we need to add.
                
                # The windows spanning i in the new string are those that include the 
                # adjacency of word[i-1] and word[i+1].
                # In the new string, these windows start at k_new such that 
                # k_new <= i-1 and k_new+n-1 >= i.
                # So k_new in [i-n+1, i].
                
                # Let's just check the specific windows that change.
                # The windows that *were* valid and *are* valid but unchanged don't need update.
                # We decremented the old ones spanning i.
                # We must increment the new ones spanning the gap.
                
                # New window content:
                # It consists of word[k_new : i] + word[i+1 : k_new+n+1]
                # Note: k_new is index in NEW string.
                # The part before gap: new[0:i] is word[0:i].
                # The part after gap: new[i:] is word[i+1:].
                
                # Window new[k_new : k_new+n]
                # If k_new + n <= i: entirely before gap. (Already handled, no change)
                # If k_new >= i: entirely after gap. (Already handled, no change)
                # If k_new < i < k_new + n: spans gap.
                
                if k_new < i and k_new + n > i:
                    # This window spans the gap.
                    # Extract it
                    w_new = tuple(word[k_new:i] + word[i+1:k_new+n+1])
                    if w_new in perm_to_idx:
                        temp_counts[perm_to_idx[w_new]] += 1
            
            # Check validity
            if all(c > 0 for c in temp_counts):
                # Valid deletion!
                word.pop(i)
                counts = temp_counts
                improved = True
                break # Restart loop with new length
                
    return "".join(map(str, word))
