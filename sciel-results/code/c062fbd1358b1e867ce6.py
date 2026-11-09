import time
import random

# Precompute helper masks
_sum_valid = [0] * 129
for _x in range(1, 129):
    _m = 0
    for _a in range(1, min(128 - _x + 1, 129)):
        _m |= (1 << _a)
    _sum_valid[_x] = _m

# Precompute for each a, the bitmask of positions b > a (for quick "b above a" checks)
_bits_above = [0] * 129
for _a in range(129):
    # bits_above[a] has bits a+1, a+2, ... set
    _m = 0
    for _b in range(_a + 1, 129):
        _m |= (1 << _b)
    _bits_above[_a] = _m

def _can_add_bitmask(mask, x):
    """
    Check if x can be added to the set represented by bitmask without creating a forbidden triple.
    Uses bit-parallel operations for speed.
    """
    # Part 1: Check triples (a, b, x) where a,b in S, a+x in S, b+x in S, a+b in S or a+b==x
    # valid_a: bits a where a in S and a+x in S and a <= 128-x
    valid_a = (mask >> x) & mask & _sum_valid[x]
    
    va = valid_a
    while va:
        a_bit = va & -va
        a = a_bit.bit_length() - 1
        va ^= a_bit
        # b must be in valid_a, b > a
        b_mask = valid_a & _bits_above[a]
        if b_mask == 0:
            continue
        # Check if any a+b is in S: (mask >> a) has bit b set iff mask has bit (a+b) set
        if ((mask >> a) & b_mask) != 0:
            return False
        # Check if a+b == x: b = x-a
        b_val = x - a
        if b_val > a and b_val <= 128:
            if (b_mask >> b_val) & 1:
                return False
    
    # Part 2: Check if x completes a sum: a+b==x with a,b in S, and exists c in S\{a,b} with a+c in S and b+c in S
    for a in range(1, x // 2 + 1):
        b = x - a
        if b <= a or b > 128:
            continue
        if not ((mask >> a) & 1) or not ((mask >> b) & 1):
            continue
        # c must be in S, c != a, c != b, a+c in S, b+c in S
        # a+c in S: bit (a+c) in mask. (mask >> a) has bit c set iff mask has bit (a+c) set
        # b+c in S: (mask >> b) has bit c set iff mask has bit (b+c) set
        # c range: 1 <= c <= 128 - b (since b >= a, 128-b is the tighter bound)
        max_c = 128 - b
        if max_c < 1:
            continue
        # c_mask: bits c where c in S, c != a, c != b, c <= max_c
        c_mask = mask & _sum_valid[b]  # c in S and c <= 128-b
        c_mask &= ~((1 << a) | (1 << b))  # exclude a and b
        if c_mask == 0:
            continue
        # Check if exists c in c_mask with (mask >> a) & c_mask != 0 AND (mask >> b) & c_mask != 0
        # This means: exists c such that a+c in S AND b+c in S
        # We need the intersection: c in c_mask AND a+c in S AND b+c in S
        # (mask >> a) & c_mask gives c where a+c in S
        # (mask >> b) & c_mask gives c where b+c in S
        # Intersection: ((mask >> a) & (mask >> b) & c_mask) != 0
        if ((mask >> a) & (mask >> b) & c_mask) != 0:
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
    deadline = start_time + 85

    def time_left():
        return deadline - time.time()

    best_mask = 0
    best_size = 0
    rng = random.Random(42)

    def mask_size(mask):
        return bin(mask).count('1')

    def greedy_fill(mask, order, time_limit=10):
        """Greedily add elements from order to mask if valid."""
        cur = mask
        for x in order:
            if time.time() > time_limit:
                break
            if (cur >> x) & 1:
                continue
            if _can_add_bitmask(cur, x):
                cur |= (1 << x)
        return cur

    def local_search(mask, max_iter=200000, time_limit=15):
        """Local search: try additions, then swaps."""
        cur = mask
        cur_size = mask_size(cur)
        no_improve = 0
        iterations = 0
        
        while time_left() > 5 and iterations < max_iter:
            iterations += 1
            made_progress = False
            
            # Try to add an element
            for x in range(1, 129):
                if (cur >> x) & 1:
                    continue
                if _can_add_bitmask(cur, x):
                    cur |= (1 << x)
                    cur_size += 1
                    made_progress = True
                    break
            
            if made_progress:
                no_improve = 0
                continue
            
            # Try single swaps
            elems = [i for i in range(1, 129) if (cur >> i) & 1]
            found_swap = False
            for r in elems:
                if time_left() < 5:
                    break
                without = cur & ~(1 << r)
                best_a = -1
                for a in range(1, 129):
                    if (without >> a) & 1:
                        continue
                    if _can_add_bitmask(without, a):
                        test_mask = without | (1 << a)
                        # Check if we can add more after swap
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
                        if a > r:
                            if best_a == -1 or a > best_a:
                                best_a = a
                if best_a != -1:
                    cur = without | (1 << best_a)
                    cur_size = mask_size(cur)
                    made_progress = True
                    found_swap = True
                    break
            
            if made_progress:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 5:
                    break
        
        return cur, cur_size

    # Strategy 1: Upper half base + greedy lower
    base_mask = 0
    for i in range(65, 129):
        base_mask |= (1 << i)
    
    lower_elements = list(range(1, 65))
    
    # Natural order
    cur_mask = greedy_fill(base_mask, lower_elements, deadline - 5)
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    if time_left() > 20:
        cur_mask, cur_size = local_search(cur_mask, max_iter=100000, time_limit=deadline - 10)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    # Random orderings
    for trial in range(50):
        if time_left() < 15:
            break
        order = lower_elements[:]
        rng.shuffle(order)
        cur_mask = greedy_fill(base_mask, order, deadline - 5)
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        if time_left() > 20:
            cur_mask, cur_size = local_search(cur_mask, max_iter=50000, time_limit=deadline - 10)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 2: Different base ranges
    for start in range(50, 65):
        if time_left() < 10:
            break
        base_mask = 0
        for i in range(start, 129):
            base_mask |= (1 << i)
        
        candidates = list(range(1, start))
        rng.shuffle(candidates)
        
        cur_mask = greedy_fill(base_mask, candidates, deadline - 5)
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        if time_left() > 15:
            cur_mask, cur_size = local_search(cur_mask, max_iter=80000, time_limit=deadline - 10)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 3: Odds/evens
    for pattern in [1, 2]:
        if time_left() < 10:
            break
        cur_mask = 0
        for i in range(pattern, 129, 2):
            cur_mask |= (1 << i)
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        if time_left() > 15:
            cur_mask, cur_size = local_search(cur_mask, max_iter=80000, time_limit=deadline - 10)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 4: Aggressive local search with removals
    if time_left() > 15:
        current_mask = best_mask
        current_size = best_size
        improved = True
        while time_left() > 8:
            improved = False
            elems = [i for i in range(1, 129) if (current_mask >> i) & 1]
            for r in elems:
                if time_left() < 8:
                    break
                without = current_mask & ~(1 << r)
                for a in range(1, 129):
                    if (without >> a) & 1:
                        continue
                    if _can_add_bitmask(without, a):
                        new_mask = without | (1 << a)
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
    
    # Final polish
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
    
    if not _is_valid_bitmask(best_mask):
        result = list(range(65, 129))
    
    return [int(x) for x in result]
