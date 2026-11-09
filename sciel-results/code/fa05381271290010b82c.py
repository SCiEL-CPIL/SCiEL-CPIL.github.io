import time
import random

def _can_add_bitmask(mask, x):
    """
    Check if x can be added to the set represented by bitmask without creating a forbidden triple.
    Forbidden triple: distinct a, b, c in S_new such that a+b, a+c, b+c in S_new.
    If x is new, any new triple must involve x.
    """
    # Check triples involving x: (a, b, x)
    # Sums: a+b, a+x, b+x must be in S_set U {x}.
    # a+x > x, b+x > x (since a,b >= 1), so a+x, b+x must be in S_set.
    # a+b in S_set or a+b == x.
    
    # For each a in S where a+x in S, check all b in S where b+x in S
    # Then check if a+b in S or a+b == x
    # Optimize: iterate a from 1 to x-1 (since a+x <= 128 means a <= 128-x)
    # and b from 1 to x-1 (since b+x <= 128 means b <= 128-x)
    
    max_val = 128 - x
    if max_val < 1:
        # No a or b can satisfy a+x <= 128, so no triple involving x as the third element
        pass
    else:
        # Check pairs (a,b) with a < b, both <= max_val, both in S, a+x in S, b+x in S
        # and a+b in S or a+b == x
        for a in range(1, max_val + 1):
            if not ((mask >> a) & 1):
                continue
            xa = x + a
            if not ((mask >> xa) & 1):
                continue
            # Now check b > a, b <= max_val
            for b in range(a + 1, max_val + 1):
                if not ((mask >> b) & 1):
                    continue
                xb = x + b
                if not ((mask >> xb) & 1):
                    continue
                ab = a + b
                if ab == x:
                    return False
                if ab <= 128 and ((mask >> ab) & 1):
                    return False
    
    # Check triples in S that become forbidden because x completes a sum
    # Triple (a, b, c) in S. One sum must be x.
    # If a+b == x, need a+c in S and b+c in S for some c != a,b
    # Find pairs (a, b) in S such that a + b == x
    for a in range(1, x // 2 + 1):
        b = x - a
        if b <= a or b > 128:
            continue
        if not ((mask >> a) & 1) or not ((mask >> b) & 1):
            continue
        # Check if there exists c in S, c != a, c != b, such that a+c in S and b+c in S
        # c must satisfy: a+c <= 128, b+c <= 128, so c <= 128 - max(a,b) = 128 - b
        # Also c >= 1, c != a, c != b
        max_c = 128 - b
        for c in range(1, max_c + 1):
            if c == a or c == b:
                continue
            if not ((mask >> c) & 1):
                continue
            ac = a + c
            bc = b + c
            if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                return False
    
    return True

def _is_valid_bitmask(mask):
    """Check if the set represented by bitmask is valid (no forbidden triple)."""
    elems = []
    for i in range(1, 129):
        if (mask >> i) & 1:
            elems.append(i)
    
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i + 1, n):
            b = elems[j]
            ab = a + b
            if ab > 128:
                continue
            if not ((mask >> ab) & 1):
                continue
            for k in range(j + 1, n):
                c = elems[k]
                ac = a + c
                bc = b + c
                if ac > 128 or bc > 128:
                    continue
                if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    def time_left():
        return deadline - time.time()
    
    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    def mask_size(mask):
        return bin(mask).count('1')
    
    def try_improve(mask, deadline, max_iterations=100000):
        """Local search: try to add elements, then try swaps."""
        current_mask = mask
        current_size = mask_size(current_mask)
        improved = True
        iterations = 0
        no_improve_count = 0
        
        while time_left() > 5 and iterations < max_iterations:
            iterations += 1
            made_progress = False
            
            # 1. Try to add an element (greedy)
            for x in range(1, 129):
                if (current_mask >> x) & 1:
                    continue
                if _can_add_bitmask(current_mask, x):
                    current_mask |= (1 << x)
                    current_size += 1
                    made_progress = True
                    break
            
            if made_progress:
                no_improve_count = 0
                continue
            
            # 2. Try single swaps: remove r, add a (equal or better size)
            elems = [i for i in range(1, 129) if (current_mask >> i) & 1]
            found_swap = False
            for r in elems:
                if time_left() < 5:
                    break
                without_mask = current_mask & ~(1 << r)
                # Try to find a better replacement
                best_a = -1
                for a in range(1, 129):
                    if (without_mask >> a) & 1:
                        continue
                    if _can_add_bitmask(without_mask, a):
                        # Check if this swap helps: try to add more elements after swap
                        test_mask = without_mask | (1 << a)
                        # Quick check: can we add any more elements to test_mask?
                        can_add_more = False
                        for x2 in range(1, 129):
                            if (test_mask >> x2) & 1:
                                continue
                            if _can_add_bitmask(test_mask, x2):
                                can_add_more = True
                                break
                        if can_add_more:
                            best_a = a
                            break
                        # Even if we can't add more immediately, accept if a > r (higher value might help)
                        if a > r:
                            if best_a == -1 or a > best_a:
                                best_a = a
                if best_a != -1:
                    current_mask = without_mask | (1 << best_a)
                    made_progress = True
                    found_swap = True
                    break
            
            if made_progress:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
        
        return current_mask, current_size
    
    # Strategy 1: Start with upper half, greedily add lower elements
    base_mask = 0
    for i in range(65, 129):
        base_mask |= (1 << i)
    
    # Try multiple orderings for adding lower elements
    lower_elements = list(range(1, 65))
    
    # Natural order
    cur_mask = base_mask
    for x in lower_elements:
        if time_left() < 10:
            break
        if _can_add_bitmask(cur_mask, x):
            cur_mask |= (1 << x)
    
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Local search from this result
    if time_left() > 20:
        cur_mask, cur_size = try_improve(cur_mask, deadline)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    # Random orderings
    for trial in range(30):
        if time_left() < 15:
            break
        order = lower_elements[:]
        rng.shuffle(order)
        cur_mask = base_mask
        for x in order:
            if time_left() < 10:
                break
            if _can_add_bitmask(cur_mask, x):
                cur_mask |= (1 << x)
        
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        # Local search
        if time_left() > 20:
            cur_mask, cur_size = try_improve(cur_mask, deadline, max_iterations=50000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 2: Try different base ranges
    for start in range(50, 65):
        if time_left() < 10:
            break
        base_mask = 0
        for i in range(start, 129):
            base_mask |= (1 << i)
        
        candidates = list(range(1, start))
        rng.shuffle(candidates)
        
        cur_mask = base_mask
        for x in candidates:
            if time_left() < 8:
                break
            if _can_add_bitmask(cur_mask, x):
                cur_mask |= (1 << x)
        
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        if time_left() > 15:
            cur_mask, cur_size = try_improve(cur_mask, deadline, max_iterations=80000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 3: Try odds/evens patterns
    # Odds only
    cur_mask = 0
    for i in range(1, 129, 2):
        cur_mask |= (1 << i)
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Evens only
    cur_mask = 0
    for i in range(2, 129, 2):
        cur_mask |= (1 << i)
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Strategy 4: More aggressive local search with removals
    if time_left() > 15:
        current_mask = best_mask
        current_size = best_size
        improved = True
        while time_left() > 8:
            improved = False
            # Try removing one element and adding one (swap)
            elems = [i for i in range(1, 129) if (current_mask >> i) & 1]
            for r in elems:
                if time_left() < 8:
                    break
                without_mask = current_mask & ~(1 << r)
                # Try adding one that enables growth
                for a in range(1, 129):
                    if (without_mask >> a) & 1:
                        continue
                    if _can_add_bitmask(without_mask, a):
                        new_mask = without_mask | (1 << a)
                        # Check if we can add more after this swap
                        can_grow = False
                        for x2 in range(1, 129):
                            if (new_mask >> x2) & 1:
                                continue
                            if _can_add_bitmask(new_mask, x2):
                                can_grow = True
                                break
                        if can_grow or a > r:
                            current_mask = new_mask
                            current_size = mask_size(current_mask)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
    
    # Final polish: try to add elements in ascending order
    if time_left() > 5:
        for x in range(1, 129):
            if time_left() < 3:
                break
            if (best_mask >> x) & 1:
                continue
            if _can_add_bitmask(best_mask, x):
                best_mask |= (1 << x)
                best_size += 1
    
    # Final verification
    result = [i for i in range(1, 129) if (best_mask >> i) & 1]
    
    # Verify validity
    if not _is_valid_bitmask(best_mask):
        # Fallback to upper half
        result = list(range(65, 129))
    
    return [int(x) for x in result]
