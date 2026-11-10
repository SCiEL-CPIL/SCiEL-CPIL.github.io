import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    limit = 180
    
    # Precompute conflicts: for each a in [1, limit], for each b in [a+1, limit],
    # compute c = ab/(b-a). If c is integer, c > a, c != b, c <= limit,
    # then (a, b, c) is a forbidden triple.
    # conflicts[a] = {b: c} means if a and b are in S, then c must NOT be in S.
    
    conflicts = {}
    for a in range(1, limit + 1):
        conflicts[a] = {}
        for b in range(a + 1, limit + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > a and c != b and c <= limit:
                    conflicts[a][b] = c
    
    # Build reverse lookup: for each x, which pairs (a,b) with a<b conflict to give c=x?
    # Actually, we need: for element x, what elements y are such that there exists z in S
    # where {x,y,z} forms a triple? This is complex, so we'll keep the direct check.
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        for y in S_set:
            if y > x:
                # x < y, check conflicts[x][y]
                if y in conflicts[x] and conflicts[x][y] in S_set:
                    return False
            elif y < x:
                # y < x, check conflicts[y][x]
                if x in conflicts[y] and conflicts[y][x] in S_set:
                    return False
        return True
    
    # Multiple greedy restarts with different orderings
    best_set = set()
    best_size = 0
    
    orderings = [
        list(range(limit, 0, -1)),  # largest to smallest
        list(range(1, limit + 1)),  # smallest to largest
    ]
    
    # Build alternating ordering
    left, right = 1, limit
    alt_order = []
    while left <= right:
        alt_order.append(left)
        left += 1
        if left <= right:
            alt_order.append(right)
            right -= 1
    orderings.append(alt_order)
    
    # More diverse random orderings
    rng = random.Random(42)
    for seed in range(10):
        r = random.Random(seed * 12345 + 67890)
        order = list(range(1, limit + 1))
        r.shuffle(order)
        orderings.append(order)
    
    # Weighted orderings: bias towards numbers with fewer conflicts
    conflict_count = [0] * (limit + 1)
    for a in range(1, limit + 1):
        conflict_count[a] = len(conflicts[a])
    
    # Order by increasing conflict count (fewer conflicts first)
    by_conflict = sorted(range(1, limit + 1), key=lambda x: conflict_count[x])
    orderings.append(by_conflict)
    
    # Order by decreasing conflict count
    by_conflict_desc = sorted(range(1, limit + 1), key=lambda x: -conflict_count[x])
    orderings.append(by_conflict_desc)
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        S_set = set()
        for a in ordering:
            if can_add(a, S_set):
                S_set.add(a)
        
        # Try to add any remaining elements
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for a in range(1, limit + 1):
                if a in S_set:
                    continue
                if can_add(a, S_set):
                    S_set.add(a)
                    improved = True
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
    
    # Local search on the best set found
    S_set = set(best_set)
    
    # Iterative improvement loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(S_set)
        
        # Try adding any element not in S
        for a in range(1, limit + 1):
            if a in S_set:
                continue
            if can_add(a, S_set):
                S_set.add(a)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove one, add one different
        if time.time() < deadline - 15:
            for remove_val in list(S_set):
                S_set.remove(remove_val)
                found = False
                for a in range(1, limit + 1):
                    if a in S_set or a == remove_val:
                        continue
                    if can_add(a, S_set):
                        S_set.add(a)
                        found = True
                        break
                
                if not found:
                    S_set.add(remove_val)
                else:
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try 1-for-2 swaps: remove one, try to add two
        if time.time() < deadline - 25:
            for remove_val in list(S_set):
                S_set.remove(remove_val)
                
                # Find all elements that can be added
                addable = []
                for a in range(1, limit + 1):
                    if a in S_set or a == remove_val:
                        continue
                    if can_add(a, S_set):
                        addable.append(a)
                
                # Try pairs from addable
                found = False
                for i in range(len(addable)):
                    a1 = addable[i]
                    S_set.add(a1)
                    for j in range(i + 1, len(addable)):
                        a2 = addable[j]
                        if a2 in S_set:
                            continue
                        if can_add(a2, S_set):
                            S_set.add(a2)
                            found = True
                            break
                    if found:
                        break
                    S_set.remove(a1)
                
                if not found:
                    S_set.add(remove_val)
                else:
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try 2-for-1 swaps: remove two, add one (to free up space)
        if time.time() < deadline - 35:
            S_list = list(S_set)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    r1, r2 = S_list[i], S_list[j]
                    S_set.remove(r1)
                    S_set.remove(r2)
                    
                    found = False
                    for a in range(1, limit + 1):
                        if a in S_set or a == r1 or a == r2:
                            continue
                        if can_add(a, S_set):
                            S_set.add(a)
                            found = True
                            break
                    
                    if not found:
                        S_set.add(r1)
                        S_set.add(r2)
                    else:
                        improved = True
                        break
                
                if improved:
                    break
        
        if improved:
            continue
        
        # If no improvement, try a small random restart from current state
        if time.time() < deadline - 40:
            # Remove 2-3 random elements and try to rebuild
            to_remove = random.sample(list(S_set), min(3, len(S_set)))
            for elem in to_remove:
                S_set.remove(elem)
            
            # Try to add back or add new
            improved = False
            for a in range(1, limit + 1):
                if a in S_set:
                    continue
                if can_add(a, S_set):
                    S_set.add(a)
                    improved = True
            
            if len(S_set) >= current_size:
                improved = True
    
    # Final cleanup: verify the set is valid
    final_set = set()
    for a in sorted(S_set):
        if can_add(a, final_set):
            final_set.add(a)
    
    # Try to add more
    for a in range(1, limit + 1):
        if a not in final_set and can_add(a, final_set):
            final_set.add(a)
    
    return list(final_set)
