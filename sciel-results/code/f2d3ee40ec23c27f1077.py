import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # 1. Generate the classic superpermutation
    word = _classic_construct(n)
    
    # 2. Local Search: Try to remove symbols efficiently
    if time.time() < deadline - 5:
        word = _local_search_remove(word, n, deadline)
    
    # 3. Try swap-based local search to find more improvements
    if time.time() < deadline - 5:
        word = _local_search_swap(word, n, deadline)
    
    # 4. Final pass of removals
    if time.time() < deadline - 2:
        word = _local_search_remove(word, n, deadline)
    
    return "".join(map(str, word))

def _classic_construct(n):
    """Build the classic recursive superpermutation."""
    word = [1]
    for m in range(2, n + 1):
        # Find all (m-1)-permutations in current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Greedy overlap construction
        if not perms:
            continue
            
        remaining = list(perms)
        first_p = remaining[0]
        remaining.remove(first_p)
        
        new = list(first_p) + [m] + list(first_p)
        
        while remaining:
            best_p = None
            best_overlap = -1
            best_idx = -1
            
            for i, p in enumerate(remaining):
                b = list(p) + [m] + list(p)
                max_ov = min(len(new), len(b) - 1)
                for ov in range(max_ov, -1, -1):
                    if new[len(new)-ov:] == b[:ov]:
                        if ov > best_overlap:
                            best_overlap = ov
                            best_p = p
                            best_idx = i
                        break
                
            if best_p is None:
                best_p = remaining[0]
                best_idx = 0
                best_overlap = 0
            
            remaining.pop(best_idx)
            b = list(best_p) + [m] + list(best_p)
            new += b[best_overlap:]
        
        word = new
    
    return word

def _local_search_remove(word, n, deadline):
    """Efficient local search that removes redundant symbols."""
    w = list(word)
    N = len(w)
    
    # Build count dictionary for all n-windows
    def build_counts(w, n):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if len(set(p)) == n:
                counts[p] = counts.get(p, 0) + 1
        return counts
    
    counts = build_counts(w, n)
    N = len(w)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Scan left to right, remove any safe symbol
        i = 0
        while i < len(w):
            if time.time() > deadline - 2:
                break
            
            # Check if removing w[i] is safe
            start = max(0, i - n + 1)
            end = min(i, len(w) - n)
            
            safe = True
            for k in range(start, end + 1):
                p = tuple(w[k:k+n])
                if len(set(p)) == n:
                    if counts.get(p, 0) <= 1:
                        safe = False
                        break
            
            if safe:
                # Remove w[i]
                w.pop(i)
                # Rebuild counts (expensive but correct)
                counts = build_counts(w, n)
                improved = True
                # Don't advance i, check same position again
            else:
                i += 1
    
    return w

def _local_search_swap(word, n, deadline):
    """Try swapping adjacent or nearby symbols to enable further removals."""
    w = list(word)
    
    def build_counts(w, n):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if len(set(p)) == n:
                counts[p] = counts.get(p, 0) + 1
        return counts
    
    def count_removals(w, n, counts):
        """Count how many symbols can be removed."""
        total = 0
        for i in range(len(w)):
            start = max(0, i - n + 1)
            end = min(i, len(w) - n)
            safe = True
            for k in range(start, end + 1):
                p = tuple(w[k:k+n])
                if len(set(p)) == n:
                    if counts.get(p, 0) <= 1:
                        safe = False
                        break
            if safe:
                total += 1
        return total
    
    best_w = list(w)
    best_removals = count_removals(w, n, build_counts(w, n))
    
    for _ in range(50):  # Try up to 50 random swap attempts
        if time.time() > deadline - 2:
            break
        
        # Pick a random position to try swapping with a nearby position
        i = random.randint(0, len(w) - 2)
        j = random.choice([i+1, i+2, i+3]) if i+3 < len(w) else i+1
        if j >= len(w):
            continue
        
        # Try swap
        w[i], w[j] = w[j], w[i]
        counts = build_counts(w, n)
        rem = count_removals(w, n, counts)
        
        if rem > best_removals:
            best_w = list(w)
            best_removals = rem
            # Do removals on best_w
            best_w = _local_search_remove(best_w, n, deadline)
            # Update w to continue from improved state
            w = list(best_w)
        else:
            # Undo swap
            w[i], w[j] = w[j], w[i]
    
    return best_w
