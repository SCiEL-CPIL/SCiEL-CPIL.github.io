import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin
    
    n = 256
    
    # Precompute conflicts: for each x, list of pairs (i,j) where {x,i,j} is a 3-AP
    conflicts = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        # x as first term: (x, x+d, x+2d)
        for d in range(1, (n - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if c <= n:
                conflicts[x].append((b, c))
        # x as middle term: (x-d, x, x+d)
        for d in range(1, min(x, n - x) + 1):
            a = x - d
            c = x + d
            if a >= 1 and c <= n:
                conflicts[x].append((a, c))
        # x as last term: (x-2d, x-d, x)
        for d in range(1, (x // 2) + 1):
            a = x - 2 * d
            b = x - d
            if a >= 1:
                conflicts[x].append((a, b))
    
    # Greedy: add elements in order of decreasing 1/x (i.e., ascending x)
    # But we want to maximize sum of 1/x, so prioritize small x
    selected_mask = 0  # bitmask of selected elements
    selected = []
    
    # Try adding in order 1, 2, 3, ..., 256
    for x in range(1, n+1):
        # Check if adding x creates a 3-AP
        can_add = True
        for (i, j) in conflicts[x]:
            if (selected_mask >> i) & 1 and (selected_mask >> j) & 1:
                can_add = False
                break
        if can_add:
            selected_mask |= (1 << x)
            selected.append(x)
    
    def reciprocal_sum(mask):
        s = 0.0
        for x in range(1, n+1):
            if (mask >> x) & 1:
                s += 1.0 / x
        return s
    
    best_mask = selected_mask
    best_sum = reciprocal_sum(best_mask)
    
    # Local search: try swaps
    def can_add_to_mask(mask, x):
        for (i, j) in conflicts[x]:
            if (mask >> i) & 1 and (mask >> j) & 1:
                return False
        return True
    
    # Try single swaps: remove y, add x
    # Also try adding x if not already in set
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        # Try adding any x not in set
        for x in range(1, n+1):
            if (selected_mask >> x) & 1:
                continue
            if can_add_to_mask(selected_mask, x):
                new_mask = selected_mask | (1 << x)
                new_sum = best_sum + 1.0 / x
                if new_sum > best_sum + 1e-12:
                    selected_mask = new_mask
                    best_sum = new_sum
                    improved = True
        # Try swaps: remove y, add x
        for y in range(1, n+1):
            if not ((selected_mask >> y) & 1):
                continue
            for x in range(1, n+1):
                if (selected_mask >> x) & 1:
                    continue
                # Remove y, try to add x
                temp_mask = selected_mask & ~(1 << y)
                if can_add_to_mask(temp_mask, x):
                    new_mask = temp_mask | (1 << x)
                    new_sum = best_sum - 1.0 / y + 1.0 / x
                    if new_sum > best_sum + 1e-12:
                        selected_mask = new_mask
                        best_sum = new_sum
                        improved = True
                        break
            if time.time() - start_time > DEADLINE:
                break
        if time.time() - start_time > DEADLINE:
            break
    
    # Extract result
    result = [x for x in range(1, n+1) if (selected_mask >> x) & 1]
    return result
