import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Use 85 seconds out of 90 to be safe
    deadline = start_time + 85.0
    
    # Generate all 4-subsets of [0, 16) as 16-bit masks
    masks = []
    for c in combinations(range(16), 4):
        m = 0
        for x in c:
            m |= (1 << x)
        masks.append(m)
    
    n_masks = len(masks)
    best_family = []
    best_size = 0
    
    # Helper to check if mask 'm' can be added to 'family'
    def can_add(m, family):
        # Group family members by their intersection with m
        # Intersection is a subset of m, so at most 16 groups
        groups = {}
        for a in family:
            inter = a & m
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(a)
        
        for h, group in groups.items():
            if len(group) < 2:
                continue
            # Check if any pair in this group has intersection exactly h
            # If group[i] & group[j] == h, then {group[i], group[j], m} is a sunflower
            for i in range(len(group)):
                ai = group[i]
                for j in range(i + 1, len(group)):
                    if ai & group[j] == h:
                        return False
        return True

    # Multiple randomized greedy restarts
    while time.time() < deadline:
        random.shuffle(masks)
        current_family = []
        
        for m in masks:
            if time.time() > deadline:
                break
            if can_add(m, current_family):
                current_family.append(m)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
            # If we hit the theoretical max or a very high number, we could stop earlier,
            # but we don't know the max. 250 is the cap.
            if best_size >= 250:
                break

    # Convert masks back to sorted lists of 1-based integers
    result = []
    for m in best_family:
        s = []
        for i in range(16):
            if m & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
