import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)
    best_family = []
    best_size = 0
    
    def can_add(family, x):
        """Check if x can be added to family without creating a sunflower.
        Sunflower condition: no three distinct sets have all three pairwise intersections equal.
        When adding x, check if there exist a,b in family such that a∩b = a∩x = b∩x."""
        n = len(family)
        if n < 2:
            return True
        
        # Group family members by their intersection with x
        groups = {}
        for a in family:
            s = a & x
            if s in groups:
                groups[s].append(a)
            else:
                groups[s] = [a]
        
        # For each group with intersection s, check if any two members have mutual intersection exactly s
        for s, group in groups.items():
            m = len(group)
            if m < 2:
                continue
            for i in range(m):
                ai = group[i]
                for j in range(i + 1, m):
                    aj = group[j]
                    if (ai & aj) == s:
                        return False
        return True
    
    # Random Greedy with multiple restarts
    while time.time() < deadline:
        shuffled_indices = list(range(n_total))
        random.shuffle(shuffled_indices)
        
        active_masks = []
        
        for idx in shuffled_indices:
            x = all_masks[idx]
            if can_add(active_masks, x):
                active_masks.append(x)
        
        if len(active_masks) > best_size:
            best_size = len(active_masks)
            best_family = active_masks[:]
    
    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for m in best_family:
        elem = []
        for i in range(16):
            if m & (1 << i):
                elem.append(i + 1)
        result.append(sorted(elem))
        
    return result
