"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    deadline = time.time() + 140  # Safety margin before 150s limit

    # 1. Generate standard superpermutation of length 1! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of 'new'
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # For n <= 6, the standard construction is often optimal or close to it, 
    # and local search might be overkill or risky if time is tight, 
    # but for n=7 we definitely want to try to improve.
    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for optimized local search
    # Map permutations to IDs for fast counting
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts of each permutation in the current word
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    # Check if valid initially (should be)
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    # Local Search: Repeatedly try to delete characters
    # We iterate until no more deletions are possible or deadline is reached
    # We do multiple passes to ensure local minima are escaped if possible (though simple deletion is greedy)
    # Actually, simple left-to-right or right-to-left deletion is standard. 
    # Doing it right-to-left is safer for index management if we pop.
    
    # Optimization: Pre-calculate window IDs for the initial state to avoid repeated tuple creation?
    # No, the word changes. We must recalculate.
    
    # Helper to get PID of a window defined by start and end in current word
    # But since we are deleting, the word changes.
    
    # We will iterate backwards.
    # To allow multiple passes if time permits, we can loop while improved.
    # However, one pass usually finds the local optimum for this specific greedy deletion.
    # Let's stick to one thorough pass first, then check time.
    
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        # Iterate backwards so indices remain valid after deletion
        # We go from L-1 down to 0
        # Note: L changes as we delete. We must use current len(word)
        current_L = len(word)
        for i in range(current_L - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Attempt deletion at index i
            # Affected old windows: start k in [max(0, i-n+1), min(i, current_L-n)]
            start_k_old = max(0, i - n + 1)
            end_k_old = min(i, current_L - n)
            
            # New word length is current_L - 1
            # New windows crossing the gap (where i was removed)
            # In the new word, the gap is between index i-1 and i.
            # Windows starting at k_new in [max(0, i-n), min(i-1, current_L-1-n)]
            start_k_new = max(0, i - n)
            end_k_new = min(i - 1, current_L - 1 - n)
            
            # We need to check if deleting i causes any count to drop to 0.
            # We simulate the change.
            
            # 1. Collect old PIDs to decrement
            old_pids = []
            for k in range(start_k_old, end_k_old + 1):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_pids.append(pid)
            
            # 2. Collect new PIDs to increment
            new_pids = []
            for k_new in range(start_k_new, end_k_new + 1):
                # Construct the new window
                # Part 1: word[k_new : i]
                # Part 2: word[i+1 : k_new+n+1]
                # Note: indices are from the CURRENT word (before deletion)
                w1 = word[k_new:i]
                w2 = word[i+1:k_new+n+1]
                p = tuple(w1 + w2)
                pid = perm_to_id.get(p)
                if pid is not None:
                    new_pids.append(pid)
            
            # 3. Check validity
            # We need to ensure that for every pid, count[pid] - (count in old_pids) + (count in new_pids) >= 1
            # Or simpler: 
            # Let delta[pid] = new_count - old_count
            # We check if min(counts[pid] + delta[pid]) >= 1 for all affected pids.
            
            # Since most pids are not affected, we only check the union of old_pids and new_pids.
            # But wait, if a pid is in old_pids but not new_pids, its count decreases.
            # If a pid is in new_pids but not old_pids, its count increases.
            # If a pid is in both, it might stay same or change.
            
            # We must ensure NO pid in the ENTIRE array becomes 0.
            # Only pids in (old_pids U new_pids) can change.
            # If a pid is not in this set, its count is unchanged and was >= 1, so it's safe.
            
            # So we only need to check pids in the set of touched PIDs.
            
            touched = set(old_pids) | set(new_pids)
            
            valid = True
            # Calculate net change for touched pids
            # We can just simulate the counts for these specific pids
            
            # To be safe and fast, we can just decrement/increment in a temporary dict or just check logic.
            # Logic:
            # For each pid in touched:
            #   new_count = counts[pid] - old_pids.count(pid) + new_pids.count(pid)
            #   if new_count < 1: invalid
            
            # Using .count() on lists is O(N). The lists are small (size <= n).
            # So this is O(|touched| * n). |touched| <= 2n. So O(n^2).
            
            for pid in touched:
                dec = old_pids.count(pid)
                inc = new_pids.count(pid)
                if counts[pid] - dec + inc < 1:
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                word.pop(i)
                # Update counts
                for pid in old_pids:
                    counts[pid] -= 1
                for pid in new_pids:
                    counts[pid] += 1
                improved = True
                current_L -= 1
            else:
                pass # Keep character

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
