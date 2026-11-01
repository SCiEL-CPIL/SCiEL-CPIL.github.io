import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 130.0 # Safety margin
    
    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Construct block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find max overlap with current new
            overlap = 0
            # Max possible overlap is len(block) - 1 or len(new)
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Local Search Deletion
    current = list(word)
    L = len(current)
    
    # Precompute counts
    def get_perm(w, idx, n):
        return tuple(w[idx:idx+n])
        
    perm_count = defaultdict(int)
    for i in range(L - n + 1):
        p = get_perm(current, i, n)
        perm_count[p] += 1
        
    target_perms = n
    for i in range(1, n):
        target_perms *= i
        
    # If we don't have all perms, return as is (should not happen with valid construction)
    if len(perm_count) < target_perms:
        return "".join(map(str, current))
        
    # Deletion loop
    # We restart the scan after each successful deletion to ensure we find all possible deletions
    # We can limit the number of passes to avoid infinite loops if logic is flawed, 
    # but logic is sound (length strictly decreases).
    
    while time.time() < deadline:
        deleted = False
        # Iterate from end to start or start to end?
        # Start to end is fine.
        for i in range(L - 1, -1, -1): # Try from end to start, often easier to remove redundant tails?
            if time.time() > deadline:
                break
            
            # Identify old windows containing i
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            # Store old windows and their counts
            old_windows = []
            can_delete = True
            
            for w in range(start_w, end_w + 1):
                p = get_perm(current, w, n)
                old_windows.append((w, p))
                if perm_count[p] == 1:
                    # Potential loss, need to check if it reappears in new windows
                    # But we can't decide yet
                    pass
            
            # Identify new windows formed
            # New windows are those starting at j in [i-n+1, i-1] in the NEW string?
            # Actually, let's just simulate the new windows.
            # New string is current[:i] + current[i+1:]
            # New windows spanning the gap:
            # Start j in new string such that j < i and j+n-1 >= i
            # j ranges from max(0, i-n+1) to i-1
            
            new_windows = []
            for j in range(max(0, i - n + 1), i):
                # Window in new string starts at j
                # Elements:
                # new[j] = current[j] if j < i
                # new[j+k] = current[j+k] if j+k < i else current[j+k+1]
                # Since j < i and j+n-1 >= i, the window crosses i.
                # The part before i: current[j:i]
                # The part after i: current[i+1 : i+1 + (n - (i-j))]
                # Length of first part: i - j
                # Length of second part: n - (i - j)
                
                prefix = current[j:i]
                suffix_len = n - (i - j)
                suffix = current[i+1 : i+1+suffix_len]
                
                # Check bounds
                if len(suffix) < suffix_len:
                    # This window extends beyond the end of the string?
                    # If i is near the end, this might be invalid.
                    # But we only consider windows that exist in the new string.
                    # New string length L-1.
                    # Window start j must satisfy j+n <= L-1.
                    if j + n > L - 1:
                        continue
                p = tuple(prefix + suffix)
                new_windows.append(p)
            
            # Check validity
            # We lose the old windows.
            # We gain the new windows.
            # If any old window had count 1, and it is NOT in new_windows, we lose it.
            # Note: A permutation might appear in multiple old windows.
            # We need to check if the count drops to 0.
            
            # Simpler:
            # Decrement counts for old_windows.
            # Increment counts for new_windows.
            # Check if any count is 0.
            # If yes, revert.
            
            # To avoid modifying perm_count repeatedly if we revert, we can use a local dict or just check.
            # Since n is small, let's just check.
            
            # Build a set of new windows for quick lookup?
            # Or just count.
            
            # Let's do the decrement/increment check carefully.
            # We only care if a count hits 0.
            
            # 1. Check old windows with count 1.
            # If an old window P has count 1, it MUST be in new_windows to survive.
            # If P is in new_windows, its count becomes 1 (from old) - 1 (removed) + 1 (added) = 1?
            # Wait.
            # If P is in old_windows (count 1) and P is in new_windows:
            # The specific instance in old is removed.
            # The specific instance in new is added.
            # Does the new instance correspond to the same permutation P? Yes.
            # So count stays 1?
            # Not necessarily. The new window is a DIFFERENT position.
            # But the permutation P is the same.
            # So if P was count 1, and we remove that 1, count becomes 0.
            # Then we add the new window which is P. Count becomes 1.
            # So it survives.
            
            # So:
            # For each old window P with count 1:
            #   If P is in new_windows: OK.
            #   Else: FAIL.
            
            # Also, what if P had count > 1?
            # It drops by 1. If it was 1, it would have been caught above.
            # So we only need to check old windows with count 1.
            
            # And we don't need to worry about new windows creating duplicates that affect validity?
            # No, validity is just "count > 0".
            
            # So the check is:
            # For every old window P with perm_count[P] == 1:
            #   If P not in new_windows:
            #       can_delete = False
            #       break
            
            # This is O(n) check.
            
            new_set = set(new_windows)
            
            for w, p in old_windows:
                if perm_count[p] == 1:
                    if p not in new_set:
                        can_delete = False
                        break
                        
            if can_delete:
                # Perform deletion
                # Update perm_count
                for w, p in old_windows:
                    perm_count[p] -= 1
                    if perm_count[p] == 0:
                        del perm_count[p]
                        
                for p in new_windows:
                    perm_count[p] += 1
                    
                current = current[:i] + current[i+1:]
                L -= 1
                deleted = True
                break # Restart scan
                
        if not deleted:
            break # No more deletions possible
            
    return "".join(map(str, current))
