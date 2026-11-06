import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0
    n = 256
    
    # Precompute 3AP conflict masks: conflict[i] is a bitmask of j such that {i,j} is part of a 3AP
    # For each pair (a,b) with a < b, they form a 3AP if:
    #   c = 2*b - a exists (c > b), OR
    #   c = 2*a - b exists (c < a), OR  
    #   a is the middle: a = (b+c)/2 i.e. 2*a = b+c
    # Actually, a pair {a,b} conflicts if there exists c != a,b such that {a,b,c} is a 3AP.
    # A 3AP is {x, x+d, x+2d} for d > 0.
    # So {a,b} is in a 3AP iff:
    #   a < b and 2*b - a <= n and 2*b - a != a and 2*b - a != b (i.e. 2*b-a > b)
    #   OR a > b and 2*a - b <= n and 2*a - b != a and 2*a - b != b
    #   OR |a-b| is even and (a+b)/2 is an integer in [1,n] and (a+b)/2 != a and (a+b)/2 != b
    # Simplify: {a,b} with a<b conflicts if:
    #   2*b - a <= n (c = 2*b-a > b, so c exists in range)
    #   OR 2*a - b >= 1 (c = 2*a-b < a, so c exists in range)  
    #   OR (a+b) % 2 == 0 and 1 <= (a+b)//2 <= n and (a+b)//2 != a and (a+b)//2 != b
    # Note: if 2*b-a <= n, then c=2*b-a is valid. If 2*a-b >= 1, c=2*a-b is valid.
    # The middle case: (a+b)/2 is between a and b, so it's different from both.
    
    # For bitmask approach: represent set S as a bitmask where bit i means (i+1) is in S
    # Precompute for each value v (1..n), a bitmask conflict_mask[v] where bit u is set
    # if {v, u} forms a 3AP with some third element.
    
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            # Check if {a, b} is part of a 3AP
            # Case 1: c = 2*b - a (c > b)
            c = 2 * b - a
            if c <= n:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                continue
            # Case 2: c = 2*a - b (c < a)
            c = 2 * a - b
            if c >= 1:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                continue
            # Case 3: middle element m = (a+b)/2
            if (a + b) % 2 == 0:
                m = (a + b) // 2
                if 1 <= m <= n and m != a and m != b:
                    conflict_mask[a] |= (1 << b)
                    conflict_mask[b] |= (1 << a)
    
    def is_valid_addition_bitmask(mask, y):
        """Check if adding y (1-indexed) to set represented by mask creates a 3AP."""
        bit_y = 1 << y
        # y conflicts with any existing element a if conflict_mask[y] has bit a set
        # i.e., if mask & conflict_mask[y] != 0
        return (mask & conflict_mask[y]) == 0
    
    def is_valid_swap_bitmask(mask, x, y):
        """Check if (S - {x}) U {y} is 3-free, where S is represented by mask."""
        # Remove x from mask
        mask_no_x = mask & ~(1 << x)
        # Check if y can be added to mask_no_x
        bit_y = 1 << y
        return (mask_no_x & conflict_mask[y]) == 0
    
    def score_bitmask(mask):
        s = 0.0
        for x in range(1, n + 1):
            if mask & (1 << x):
                s += 1.0 / x
        return s
    
    # 1. Greedy Init using bitmask
    mask = 0
    for x in range(1, n + 1):
        if is_valid_addition_bitmask(mask, x):
            mask |= (1 << x)
    
    best_score = score_bitmask(mask)
    best_mask = mask
    
    def local_search(mask):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Add any valid element
            for y in range(1, n + 1):
                if time.time() > deadline:
                    break
                if not (mask & (1 << y)):
                    if is_valid_addition_bitmask(mask, y):
                        mask |= (1 << y)
                        improved = True
            
            # Pass 2: Single swap - replace x with y where 1/y > 1/x (i.e., y < x)
            # For each missing y in ascending order, try to swap with some x > y in S
            for y in range(1, n + 1):
                if time.time() > deadline:
                    break
                if mask & (1 << y):
                    continue
                # Try to find x > y in S such that swap is valid
                for x in range(y + 1, n + 1):
                    if mask & (1 << x):
                        if is_valid_swap_bitmask(mask, x, y):
                            mask &= ~(1 << x)
                            mask |= (1 << y)
                            improved = True
                            break  # Move to next y
            
            # Pass 3: 2-swap - remove two elements, add one or two different elements
            # This helps escape local optima. Try: remove x1, x2 from S, add y1, y2 (y1,y2 not in S)
            # where the net change in score is positive.
            # To keep it tractable, try removing one large x and adding two smaller y1, y2.
            # Or remove two and add one.
            # Let's try: for each pair (x1, x2) in S with x1 < x2, try to find y not in S
            # such that (S - {x1,x2}) U {y} is valid and 1/y > 1/x1 + 1/x2
            # This is expensive, so limit the search.
            if time.time() < deadline - 2:
                S_elements = [x for x in range(1, n + 1) if mask & (1 << x)]
                # Only try swaps involving the largest elements
                S_desc = sorted(S_elements, reverse=True)
                missing = [y for y in range(1, n + 1) if not (mask & (1 << y))]
                
                for x1 in S_desc[:10]:  # Limit to top 10 largest
                    if time.time() > deadline:
                        break
                    for x2 in S_desc[:10]:
                        if x2 >= x1:
                            continue
                        # Try to replace {x1, x2} with a single y
                        for y in missing:
                            if time.time() > deadline:
                                break
                            if y >= x1:
                                break
                            # Check if (S - {x1,x2}) U {y} is valid
                            mask_test = mask & ~(1 << x1) & ~(1 << x2)
                            if is_valid_addition_bitmask(mask_test, y):
                                # Check score improvement
                                if 1.0 / y > 1.0 / x1 + 1.0 / x2:
                                    mask = mask_test | (1 << y)
                                    improved = True
                                    S_elements = [x for x in range(1, n + 1) if mask & (1 << x)]
                                    S_desc = sorted(S_elements, reverse=True)
                                    missing = [y for y in range(1, n + 1) if not (mask & (1 << y))]
                                    break
                        if improved:
                            break
                    if improved:
                        break
                        
        return mask
    
    mask = local_search(mask)
    sc = score_bitmask(mask)
    if sc > best_score:
        best_score = sc
        best_mask = mask
    
    # 2. Perturbation / Restart
    while time.time() < deadline - 3.0:
        S_current = [x for x in range(1, n + 1) if best_mask & (1 << x)]
        
        # Strategy: remove 1-3 random elements, prefer larger ones
        num_remove = random.randint(1, 3)
        # Weight toward larger elements
        weights = [1.0 / x for x in S_current]
        total_w = sum(weights)
        indices = list(range(len(S_current)))
        
        removed = set()
        for _ in range(min(num_remove, len(S_current))):
            if not indices:
                break
            r = random.random() * total_w
            cum = 0
            for i, idx in enumerate(indices):
                cum += weights[idx]
                if r <= cum:
                    removed.add(S_current[idx])
                    indices.remove(idx)
                    total_w -= weights[idx]
                    break
        
        mask_current = best_mask
        for x in removed:
            mask_current &= ~(1 << x)
        
        S_new = local_search(mask_current)
        score_new = score_bitmask(S_new)
        if score_new > best_score:
            best_score = score_new
            best_mask = S_new
        
        # Also try a fresh greedy start with random order
        if time.time() < deadline - 5.0:
            order = list(range(1, n + 1))
            random.shuffle(order)
            mask_fresh = 0
            for x in order:
                if is_valid_addition_bitmask(mask_fresh, x):
                    mask_fresh |= (1 << x)
            mask_fresh = local_search(mask_fresh)
            score_fresh = score_bitmask(mask_fresh)
            if score_fresh > best_score:
                best_score = score_fresh
                best_mask = mask_fresh
    
    return sorted([x for x in range(1, n + 1) if best_mask & (1 << x)])
