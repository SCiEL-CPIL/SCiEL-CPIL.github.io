import time
import itertools
import heapq

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Generate classic superpermutation
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
            # Overlap logic: find max overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n <= 4:
        return "".join(map(str, word))

    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts of each permutation in the current word
    perm_counts = {p: 0 for p in all_perms}
    
    # Initial count pass
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in perm_counts:
            perm_counts[w] += 1
    
    def is_safe_delete(i):
        """Check if deleting position i is safe (no permutation with count 1 is destroyed)."""
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        for s in range(s_min, s_max + 1):
            w = tuple(word[s:s+n])
            if w in perm_counts and perm_counts[w] == 1:
                return False
        return True
    
    def delete_position(i):
        """Delete position i and update perm_counts accordingly."""
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        
        # Decrement counts of windows that are removed
        for s in range(s_min, s_max + 1):
            w = tuple(word[s:s+n])
            if w in perm_counts:
                perm_counts[w] -= 1
        
        # Remove the element
        word.pop(i)
        
        # Increment counts of new windows formed by the shift
        new_s_min = max(0, i - n + 1)
        new_s_max = min(i, len(word) - n)
        
        if new_s_min <= new_s_max:
            for s in range(new_s_min, new_s_max + 1):
                w = tuple(word[s:s+n])
                if w in perm_counts:
                    perm_counts[w] += 1
    
    def get_min_count_at(i):
        """Get the minimum permutation count among windows affected by deleting position i."""
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        min_count = float('inf')
        for s in range(s_min, s_max + 1):
            w = tuple(word[s:s+n])
            if w in perm_counts:
                min_count = min(min_count, perm_counts[w])
        return min_count
    
    # Phase 1: Standard left-to-right greedy deletion
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            if is_safe_delete(i):
                delete_position(i)
                changed = True
                # Do NOT increment i
            else:
                i += 1
    
    # Phase 2: Right-to-left greedy deletion
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                return "".join(map(str, word))
            if is_safe_delete(i):
                delete_position(i)
                changed = True
                # Do NOT decrement i
            else:
                i -= 1
    
    # Phase 3: Priority-based deletion - always delete the position with highest min_count
    # among safe positions, to maximize future deletion opportunities
    if time.time() < deadline:
        # Build priority queue: (neg_min_count, index) so we pop highest min_count first
        # We'll use a simple approach: repeatedly find the best position
        while time.time() < deadline:
            L = len(word)
            if L == 0:
                break
            
            best_i = -1
            best_min_count = -1
            
            for i in range(L):
                if is_safe_delete(i):
                    mc = get_min_count_at(i)
                    if mc > best_min_count:
                        best_min_count = mc
                        best_i = i
            
            if best_i == -1:
                break
            
            delete_position(best_i)
    
    return "".join(map(str, word))
