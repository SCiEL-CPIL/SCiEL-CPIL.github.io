import time
import math
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Step 1: Generate classic superpermutation
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # If n is small, just return
    if n <= 5:
        return "".join(map(str, word))
        
    # Step 2: Greedy Deletion
    target_perms = math.factorial(n)
    
    # Initialize counts
    counts = Counter()
    seen_set = set()
    for i in range(len(word) - n + 1):
        perm = tuple(word[i:i+n])
        counts[perm] += 1
        seen_set.add(perm)
        
    # If already valid, we can start deleting
    # We want to reduce length while keeping len(seen_set) == target_perms
    
    while time.time() < deadline:
        improved = False
        # Iterate through all possible deletion positions
        # We restart the scan after each successful deletion to ensure we catch new opportunities
        # To avoid infinite loops or excessive time, we limit the number of passes
        # But since each deletion reduces length, the number of deletions is bounded by initial length
        
        for i in range(len(word)):
            if time.time() > deadline:
                break
                
            # Simulate deletion of word[i]
            # Windows to remove: start k in [i-n+1, i]
            remove_perms = []
            for k in range(max(0, i - n + 1), i + 1):
                if k + n <= len(word):
                    p = tuple(word[k:k+n])
                    remove_perms.append(p)
            
            # Windows to add: start k in [max(0, i-n+1), i-1]
            # New window is word[k:i] + word[i+1:k+n]
            add_perms = []
            for k in range(max(0, i - n + 1), i):
                if k + n <= len(word): # In new word, length is len(word)-1
                    # Check if this window is valid in new word context
                    # The window in new word starting at k ends at k+n-1
                    # In old word indices, it covers k..i-1 and i+1..k+n
                    # We need k+n <= len(word) to have enough chars in old word?
                    # New word length is L-1.
                    # Window start k must be < L-1 - n + 1 = L-n.
                    # So k <= L-n-1.
                    # My loop range is up to i-1.
                    # If i is near the end, we might not have enough chars.
                    # Let's just construct the tuple and check length.
                    w1 = word[k:i]
                    w2 = word[i+1:k+n]
                    if len(w1) + len(w2) == n:
                        p = tuple(w1 + w2)
                        add_perms.append(p)
            
            # Apply changes
            for p in remove_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    seen_set.discard(p)
            
            for p in add_perms:
                counts[p] += 1
                seen_set.add(p)
                
            # Check validity
            if len(seen_set) == target_perms:
                # Valid deletion
                word = word[:i] + word[i+1:]
                improved = True
                break # Restart scan
            else:
                # Revert changes
                for p in remove_perms:
                    counts[p] += 1
                    seen_set.add(p)
                
                for p in add_perms:
                    counts[p] -= 1
                    if counts[p] == 0:
                        seen_set.discard(p)
        
        if not improved:
            break
            
    return "".join(map(str, word))
