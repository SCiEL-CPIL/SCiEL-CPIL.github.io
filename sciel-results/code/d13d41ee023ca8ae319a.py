import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # 1. Generate the classic superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Greedy overlap construction
        # We need to order 'perms' to maximize overlaps
        # Start with an empty new word
        new = []
        remaining = list(perms)
        
        if not remaining:
            word = new
            continue
            
        # Pick the first permutation. We can try a few candidates.
        # Let's pick the one that appears first in the original word to maintain some structure,
        # or just the first one.
        # Actually, the choice of the first one matters.
        # We can try all candidates for the first one if n is small, but for n=7, 720 is too many.
        # Let's just pick the first one from the list.
        first_p = remaining[0]
        remaining.remove(first_p)
        
        block = list(first_p) + [m] + list(first_p)
        new = block
        
        while remaining:
            # Find the permutation in remaining that has the max overlap with new
            best_p = None
            best_overlap = -1
            best_idx = -1
            
            for i, p in enumerate(remaining):
                b = list(p) + [m] + list(p)
                # Calculate overlap between new and b
                # Max possible overlap is len(b) - 1
                max_ov = min(len(new), len(b) - 1)
                for ov in range(max_ov, -1, -1):
                    if new[len(new)-ov:] == b[:ov]:
                        if ov > best_overlap:
                            best_overlap = ov
                            best_p = p
                            best_idx = i
                        break # No need to check smaller overlaps for this p
                
            if best_p is None:
                # Should not happen
                best_p = remaining[0]
                best_idx = 0
                best_overlap = 0
            
            remaining.pop(best_idx)
            b = list(best_p) + [m] + list(best_p)
            new += b[best_overlap:]
            
        word = new
        
        if time.time() > deadline:
            break
    
    # 2. Local Search: Try to remove symbols
    # This step can be expensive, so we limit it.
    if time.time() < deadline - 10:
        word = _local_search_remove(word, n, deadline)
        
    return "".join(map(str, word))

def _local_search_remove(word, n, deadline):
    # Convert to list for mutation
    w = list(word)
    
    # Count permutations
    def get_perm_count(w, n):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if len(set(p)) == n:
                counts[p] = counts.get(p, 0) + 1
        return counts
    
    counts = get_perm_count(w, n)
    all_perms = set(counts.keys())
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Iterate through positions
        # We can randomize the order of positions to escape local minima
        positions = list(range(len(w)))
        random.shuffle(positions)
        
        for i in positions:
            if time.time() > deadline - 5:
                break
            if i >= len(w):
                continue
            
            # Check if removing w[i] is safe
            # Windows affected: i-n+1 to i
            start = max(0, i - n + 1)
            end = min(i, len(w) - n)
            
            safe = True
            affected_perms = []
            for k in range(start, end + 1):
                p = tuple(w[k:k+n])
                if len(set(p)) == n:
                    if counts.get(p, 0) <= 1:
                        safe = False
                        break
                    affected_perms.append(p)
            
            if safe:
                # Remove w[i]
                w.pop(i)
                # Update counts
                # We need to recount affected windows
                # The windows that were affected are now shifted or gone
                # It's easier to just recount everything if n is small?
                # For n=7, len(w) ~ 5900. Recounting is O(N*n) ~ 40k ops.
                # This is fast enough.
                counts = get_perm_count(w, n)
                improved = True
                # After removal, the positions shift. We break and restart the loop
                # to avoid index errors.
                break
                
    return w
