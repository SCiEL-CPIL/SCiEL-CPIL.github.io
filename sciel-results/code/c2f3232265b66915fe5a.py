import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # --- 1. Generate Baseline Recursive Superpermutation ---
    def recursive_superperm(n_val):
        word = [1]
        for m in range(2, n_val + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    baseline_word = recursive_superperm(n)
    baseline_len = len(baseline_word)
    
    if n <= 4:
        return "".join(map(str, baseline_word))

    S = baseline_word[:]
    best_S = S[:]
    best_len = len(S)

    # --- 2. Efficient Local Search ---
    
    def build_counts(word, n_val):
        counts = {}
        for i in range(len(word) - n_val + 1):
            p = tuple(word[i:i+n_val])
            counts[p] = counts.get(p, 0) + 1
        return counts

    counts = build_counts(S, n)
    
    # Pre-compute unique permutations for quick lookup? 
    # No, just check counts[p] == 1 on the fly.
    
    # Helper to check if removal at index i is valid
    def is_valid_removal(i, L):
        # Old windows indices: j such that j <= i < j+n
        # j ranges from max(0, i-n+1) to min(i, L-n)
        old_start = max(0, i - n + 1)
        old_end = min(i, L - n)
        
        # New windows indices in the modified string S' (length L-1)
        # The gap is at i.
        # Windows in S' that are affected are those that cross the gap.
        # In S', indices are 0..L-2.
        # A window at k in S' corresponds to S indices:
        # if k+n-1 < i: S[k:k+n] (unchanged)
        # if k >= i: S[k+1:k+1+n] (shifted)
        # if k < i < k+n: crosses gap.
        
        # The crossing windows in S' are those where k < i and k+n > i.
        # i.e., k > i-n and k < i.
        # k ranges from max(0, i-n+1) to min(i-1, L-1-n)
        
        new_start = max(0, i - n + 1)
        new_end = min(i - 1, L - 1 - n)
        
        # We need to ensure no count drops to 0.
        # This only happens if a unique window (count==1) is destroyed and not recreated.
        
        # Let's collect the old windows that have count 1.
        unique_old = []
        for j in range(old_start, old_end + 1):
            p = tuple(S[j:j+n])
            if counts.get(p, 0) == 1:
                unique_old.append(p)
        
        if not unique_old:
            return True
        
        # Check if any unique_old window is recreated in the new configuration.
        # We only need to check the new windows that *could* match them.
        # Actually, we just need to check if *any* of the unique_old windows are present in the new set of windows.
        # But wait, if a unique window is destroyed, its count drops to 0.
        # If it is recreated, its count stays 1 (1 - 1 + 1 = 1).
        # So we just need to check if the set of unique_old windows is a subset of the set of new windows?
        # No. If unique window U is in old, and not in new, it's gone.
        # If U is in old and in new, it's preserved.
        
        # So: For each p in unique_old, check if p is in the new windows.
        # To do this efficiently, we can just construct the new windows and check membership.
        
        new_windows_set = set()
        for k in range(new_start, new_end + 1):
            # Construct window from S
            # Indices in S for window starting at k in S':
            # If k < i:
            #   positions k..i-1 come from S[k..i-1]
            #   positions i..k+n-1 come from S[i+1..k+n]
            # Let's just build the tuple
            w = []
            for idx in range(n):
                s_idx = k + idx
                if s_idx >= i:
                    s_idx += 1
                w.append(S[s_idx])
            new_windows_set.add(tuple(w))
            
        for p in unique_old:
            if p not in new_windows_set:
                return False
        return True

    stuck_passes = 0
    max_stuck_passes = 100 # Lower since checks are fast, but let's be safe
    
    # Directions: -1 (right to left), 1 (left to right)
    # We alternate or randomize to escape local minima?
    # For now, simple right-to-left is standard.
    
    while time.time() < deadline:
        removed_in_pass = 0
        L = len(S)
        
        # Iterate from right to left to handle index shifting naturally
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
                
            if not is_valid_removal(i, L):
                continue
            
            # Perform removal
            # 1. Update counts for old windows
            old_start = max(0, i - n + 1)
            old_end = min(i, L - n)
            
            for j in range(old_start, old_end + 1):
                p = tuple(S[j:j+n])
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # 2. Pop
            S.pop(i)
            L -= 1
            removed_in_pass += 1
            
            # 3. Update counts for new windows
            new_start = max(0, i - n + 1)
            new_end = min(i - 1, L - n)
            
            for k in range(new_start, new_end + 1):
                w = []
                for idx in range(n):
                    s_idx = k + idx
                    if s_idx >= i:
                        s_idx += 1
                    w.append(S[s_idx])
                p = tuple(w)
                counts[p] = counts.get(p, 0) + 1
            
            best_S = S[:]
            best_len = L
            
        if removed_in_pass == 0:
            stuck_passes += 1
            if stuck_passes > max_stuck_passes:
                break
            # Optional: Try a random swap to unstuck? 
            # Given time constraints and that removal is the main driver, 
            # we might just break or try a different direction.
            # Let's try a forward pass if stuck in backward pass?
            # Actually, let's just break. The structure of superperms often leads to a stable local minimum.
        else:
            stuck_passes = 0

    if best_len < baseline_len:
        return "".join(map(str, best_S))
    else:
        return "".join(map(str, baseline_word))
