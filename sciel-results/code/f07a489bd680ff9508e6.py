"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    random.seed(42)
    
    def compute_counts(chosen):
        counts = {}
        n = len(chosen)
        for i in range(n):
            a = chosen[i]
            for j in range(i, n):
                b = chosen[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    def can_add(chosen, counts, x):
        """Check if x can be added without violating B2[2] constraint."""
        n = len(chosen)
        # Check sums x + chosen[i] for i in range(n), and 2*x
        new_sums = {}
        for i in range(n):
            s = x + chosen[i]
            new_sums[s] = new_sums.get(s, 0) + 1
        s2 = 2 * x
        new_sums[s2] = new_sums.get(s2, 0) + 1
        
        for s, cnt in new_sums.items():
            if counts.get(s, 0) + cnt > 2:
                return False
        return True
    
    def try_add(chosen, counts, x):
        """Add x to chosen and update counts. Returns True if successful."""
        if not can_add(chosen, counts, x):
            return False
        n = len(chosen)
        for i in range(n):
            s = x + chosen[i]
            counts[s] = counts.get(s, 0) + 1
        s2 = 2 * x
        counts[s2] = counts.get(s2, 0) + 1
        chosen.append(x)
        return True
    
    def remove_from(chosen, counts, x):
        """Remove x from chosen and update counts."""
        idx = chosen.index(x)
        n = len(chosen)
        # Remove sums involving x
        for i in range(n):
            if i == idx:
                continue
            s = x + chosen[i]
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s2 = 2 * x
        counts[s2] -= 1
        if counts[s2] == 0:
            del counts[s2]
        chosen.pop(idx)
    
    # Phase 1: Greedy construction
    chosen = []
    counts = {}
    for x in range(1, 1001):
        try_add(chosen, counts, x)
    
    best_size = len(chosen)
    
    # Phase 2: Local search with swaps
    in_set = set(chosen)
    not_in_set = [x for x in range(1, 1001) if x not in in_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try swaps: remove one element, add another
        random.shuffle(not_in_set)
        random.shuffle(chosen)
        
        for out_x in chosen[:]:
            if time.time() > deadline:
                break
            remove_from(chosen, counts, out_x)
            in_set.discard(out_x)
            
            added = False
            for in_x in not_in_set:
                if in_x in in_set:
                    continue
                if try_add(chosen, counts, in_x):
                    in_set.add(in_x)
                    if in_x in not_in_set:
                        not_in_set.remove(in_x)
                    added = True
                    improved = True
                    break
            
            if not added:
                # Put it back
                try_add(chosen, counts, out_x)
                in_set.add(out_x)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
    
    # Phase 3: Try to add more elements
    if time.time() < deadline:
        in_set = set(chosen)
        not_in_set = [x for x in range(1, 1001) if x not in in_set]
        random.shuffle(not_in_set)
        for x in not_in_set:
            if time.time() > deadline:
                break
            if try_add(chosen, counts, x):
                in_set.add(x)
    
    if len(chosen) > best_size:
        best_size = len(chosen)
    
    return chosen
# EVOLVE-BLOCK-END
