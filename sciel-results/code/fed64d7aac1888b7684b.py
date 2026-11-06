"""Optimized solver for 3-AP-rich, 4-AP-free subsets of [1,256]."""
import time
import random

N = 256
DEADLINE = 80.0  # safety margin before 90s kill

def _build_ap_tables():
    """Precompute 3-AP and 4-AP membership info for O(1) delta evaluation."""
    # For each element x, list of 3-APs it participates in (as a, b, c with b-x = x-b = d)
    # A 3-AP is (a, a+d, a+2d) for a >= 1, d >= 1, a+2d <= N
    # We count each 3-AP once (positive difference)
    
    # 3-APs: list of (a, b, c) where b = (a+c)//2, a < b < c
    # For delta eval: if we add x, how many new 3-APs form?
    # A 3-AP containing x: x can be first, middle, or last
    # x as first: (x, x+d, x+2d) for d >= 1, x+2d <= N
    # x as middle: (x-d, x, x+d) for d >= 1, x-d >= 1
    # x as last: (x-2d, x-d, x) for d >= 1, x-2d >= 1
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    # For 4-AP detection: a set is invalid if it contains all 4 elements of any 4-AP
    
    # Strategy: For each element, precompute which 3-APs and 4-APs it's part of.
    # For delta: adding x creates a 3-AP if the other 2 elements are present.
    # For 4-AP: adding x creates a 4-AP if the other 3 elements are present.
    
    # 3-APs containing element x (as any position):
    # Represent each 3-AP by its middle element and difference: (b, d) means (b-d, b, b+d)
    # Or just list all 3-APs as tuples.
    
    # Let's precompute for each x:
    #   three_aps_with_x[x] = list of (other1, other2) such that {x, other1, other2} is a 3-AP
    #   four_aps_with_x[x] = list of (other1, other2, other3) such that {x, other1, other2, other3} is a 4-AP
    
    three_aps_with = [[] for _ in range(N+1)]
    four_aps_with = [[] for _ in range(N+1)]
    
    # Enumerate all 3-APs
    for a in range(1, N+1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            # 3-AP: (a, b, c)
            three_aps_with[a].append((b, c))
            three_aps_with[b].append((a, c))
            three_aps_with[c].append((a, b))
    
    # Enumerate all 4-APs
    for a in range(1, N+1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            four_aps_with[a].append((b, c, e))
            four_aps_with[b].append((a, c, e))
            four_aps_with[c].append((a, b, e))
            four_aps_with[e].append((a, b, c))
    
    return three_aps_with, four_aps_with

def _count_3aps(selected_mask, three_aps_with):
    """Count 3-APs in selected set using bitmask."""
    count = 0
    for x in range(1, N+1):
        if selected_mask & (1 << x):
            for (o1, o2) in three_aps_with[x]:
                if (selected_mask & (1 << o1)) and (selected_mask & (1 << o2)):
                    count += 1
    # Each 3-AP is counted 3 times (once for each element)
    return count // 3

def _has_4ap(selected_mask, four_aps_with):
    """Check if selected set contains any 4-AP."""
    for x in range(1, N+1):
        if selected_mask & (1 << x):
            for (o1, o2, o3) in four_aps_with[x]:
                if (selected_mask & (1 << o1)) and (selected_mask & (1 << o2)) and (selected_mask & (1 << o3)):
                    return True
    return False

def construct():
    start_time = time.time()
    three_aps_with, four_aps_with = _build_ap_tables()
    
    best_mask = 0
    best_count = 0
    
    def elapsed():
        return time.time() - start_time
    
    def try_add(mask, x):
        """Try adding x to mask. Return (new_mask, new_3ap_count) if valid, else None."""
        if mask & (1 << x):
            return None
        # Check 4-AP constraint
        for (o1, o2, o3) in four_aps_with[x]:
            if (mask & (1 << o1)) and (mask & (1 << o2)) and (mask & (1 << o3)):
                return None
        # Count new 3-APs formed
        new_3aps = 0
        for (o1, o2) in three_aps_with[x]:
            if (mask & (1 << o1)) and (mask & (1 << o2)):
                new_3aps += 1
        return (mask | (1 << x), new_3aps)
    
    def try_remove(mask, x, current_count):
        """Remove x from mask. Return (new_mask, new_3ap_count)."""
        if not (mask & (1 << x)):
            return None
        # Count 3-APs lost
        lost_3aps = 0
        for (o1, o2) in three_aps_with[x]:
            if (mask & (1 << o1)) and (mask & (1 << o2)):
                lost_3aps += 1
        return (mask & ~(1 << x), current_count - lost_3aps)
    
    def greedy_construct(order, start_mask=0):
        """Greedy: add elements in given order if valid and beneficial."""
        mask = start_mask
        count = 0
        for x in order:
            if elapsed() > DEADLINE:
                break
            result = try_add(mask, x)
            if result is not None:
                new_mask, new_3aps = result
                if new_3aps > 0:  # Only add if it creates at least one 3-AP
                    mask = new_mask
                    count += new_3aps
        return mask, count
    
    def hill_climb(mask, count):
        """Exhaustive single-move hill climbing: add, remove, swap."""
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            best_move = None
            best_gain = 0
            best_new_mask = mask
            best_new_count = count
            
            # Try adding each element not in mask
            for x in range(1, N+1):
                if elapsed() > DEADLINE:
                    return best_new_mask, best_new_count
                if mask & (1 << x):
                    continue
                result = try_add(mask, x)
                if result is not None:
                    new_mask, new_3aps = result
                    if new_3aps > best_gain:
                        best_gain = new_3aps
                        best_move = ('add', x)
                        best_new_mask = new_mask
                        best_new_count = count + new_3aps
            
            # Try removing each element in mask
            for x in range(1, N+1):
                if elapsed() > DEADLINE:
                    return best_new_mask, best_new_count
                if not (mask & (1 << x)):
                    continue
                result = try_remove(mask, x, count)
                if result is not None:
                    new_mask, new_count = result
                    # Removing can't increase 3-AP count directly, but may enable additions later
                    # For now, only remove if it doesn't decrease count (shouldn't happen in valid state)
                    # Actually removing always decreases or keeps same. Skip removals in pure add-only HC.
                    # But removals can enable better additions. Let's allow removals that don't hurt.
                    if new_count >= count:
                        if new_count > best_new_count:
                            best_gain = new_count - count
                            best_move = ('remove', x)
                            best_new_mask = new_mask
                            best_new_count = new_count
            
            # Try swaps: remove x, add y
            if elapsed() < DEADLINE:
                for x in range(1, N+1):
                    if elapsed() > DEADLINE:
                        return best_new_mask, best_new_count
                    if not (mask & (1 << x)):
                        continue
                    # Remove x
                    lost = 0
                    for (o1, o2) in three_aps_with[x]:
                        if (mask & (1 << o1)) and (mask & (1 << o2)):
                            lost += 1
                    temp_mask = mask & ~(1 << x)
                    temp_count = count - lost
                    
                    # Try adding y
                    for y in range(1, N+1):
                        if elapsed() > DEADLINE:
                            return best_new_mask, best_new_count
                        if temp_mask & (1 << y):
                            continue
                        # Check 4-AP
                        valid = True
                        for (o1, o2, o3) in four_aps_with[y]:
                            if (temp_mask & (1 << o1)) and (temp_mask & (1 << o2)) and (temp_mask & (1 << o3)):
                                valid = False
                                break
                        if not valid:
                            continue
                        # Count new 3-APs from adding y
                        gained = 0
                        for (o1, o2) in three_aps_with[y]:
                            if (temp_mask & (1 << o1)) and (temp_mask & (1 << o2)):
                                gained += 1
                        net = gained - lost
                        if net > best_gain:
                            best_gain = net
                            best_move = ('swap', x, y)
                            best_new_mask = temp_mask | (1 << y)
                            best_new_count = temp_count + gained
            
            if best_move is not None:
                mask = best_new_mask
                count = best_new_count
                improved = True
                if count > best_count:
                    best_count = count
                    best_mask = mask
        
        return mask, count
    
    # Multiple greedy orderings
    random.seed(42)
    orders = []
    # Natural order
    orders.append(list(range(1, N+1)))
    # Reverse order
    orders.append(list(range(N, 0, -1)))
    # Random orders
    for _ in range(10):
        o = list(range(1, N+1))
        random.shuffle(o)
        orders.append(o)
    # Order by "density" - elements with more potential 3-APs first
    density = [0] * (N+1)
    for x in range(1, N+1):
        density[x] = len(three_aps_with[x])
    orders.append(sorted(range(1, N+1), key=lambda x: -density[x]))
    orders.append(sorted(range(1, N+1), key=lambda x: density[x]))
    
    # Run greedy + hill climbing for each ordering
    for i, order in enumerate(orders):
        if elapsed() > DEADLINE:
            break
        mask, count = greedy_construct(order)
        if count > best_count:
            best_count = count
            best_mask = mask
        # Hill climb
        mask, count = hill_climb(mask, count)
        if count > best_count:
            best_count = count
            best_mask = mask
    
    # Perturbation restarts
    restart_count = 0
    while elapsed() < DEADLINE - 5:
        # Perturb best solution: randomly remove some elements
        perturbed_mask = best_mask
        perturbed_count = best_count
        num_removals = random.randint(3, 8)
        elements_in_best = [x for x in range(1, N+1) if best_mask & (1 << x)]
        if len(elements_in_best) <= num_removals:
            break
        to_remove = random.sample(elements_in_best, num_removals)
        for x in to_remove:
            result = try_remove(perturbed_mask, x, perturbed_count)
            if result is not None:
                perturbed_mask, perturbed_count = result
        
        # Re-greedy and hill climb
        mask, count = greedy_construct(list(range(1, N+1)), perturbed_mask)
        mask, count = hill_climb(mask, count)
        if count > best_count:
            best_count = count
            best_mask = mask
        restart_count += 1
    
    # Convert best_mask to sorted list
    result = [x for x in range(1, N+1) if best_mask & (1 << x)]
    
    if not result:
        # Fallback: return a known valid set
        result = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return result

# Precompute tables at module level for speed
_three_aps_with, _four_aps_with = _build_ap_tables()

# Override the functions to use precomputed tables
def construct():
    start_time = time.time()
    three_aps_with = _three_aps_with
    four_aps_with = _four_aps_with
    
    best_mask = 0
    best_count = 0
    
    def elapsed():
        return time.time() - start_time
    
    def try_add(mask, x):
        if mask & (1 << x):
            return None
        for (o1, o2, o3) in four_aps_with[x]:
            if (mask & (1 << o1)) and (mask & (1 << o2)) and (mask & (1 << o3)):
                return None
        new_3aps = 0
        for (o1, o2) in three_aps_with[x]:
            if (mask & (1 << o1)) and (mask & (1 << o2)):
                new_3aps += 1
        return (mask | (1 << x), new_3aps)
    
    def try_remove(mask, x, current_count):
        if not (mask & (1 << x)):
            return None
        lost_3aps = 0
        for (o1, o2) in three_aps_with[x]:
            if (mask & (1 << o1)) and (mask & (1 << o2)):
                lost_3aps += 1
        return (mask & ~(1 << x), current_count - lost_3aps)
    
    def greedy_construct(order, start_mask=0):
        mask = start_mask
        count = 0
        for x in order:
            if elapsed() > DEADLINE:
                break
            result = try_add(mask, x)
            if result is not None:
                new_mask, new_3aps = result
                if new_3aps > 0:
                    mask = new_mask
                    count += new_3aps
        return mask, count
    
    def hill_climb(mask, count):
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            best_gain = 0
            best_new_mask = mask
            best_new_count = count
            
            for x in range(1, N+1):
                if elapsed() > DEADLINE:
                    return best_new_mask, best_new_count
                if mask & (1 << x):
                    continue
                result = try_add(mask, x)
                if result is not None:
                    new_mask, new_3aps = result
                    if new_3aps > best_gain:
                        best_gain = new_3aps
                        best_new_mask = new_mask
                        best_new_count = count + new_3aps
            
            if best_gain > 0:
                mask = best_new_mask
                count = best_new_count
                improved = True
                if count > best_count:
                    best_count = count
                    best_mask = mask
        
        return mask, count
    
    random.seed(42)
    orders = []
    orders.append(list(range(1, N+1)))
    orders.append(list(range(N, 0, -1)))
    for _ in range(15):
        o = list(range(1, N+1))
        random.shuffle(o)
        orders.append(o)
    density = [0] * (N+1)
    for x in range(1, N+1):
        density[x] = len(three_aps_with[x])
    orders.append(sorted(range(1, N+1), key=lambda x: -density[x]))
    orders.append(sorted(range(1, N+1), key=lambda x: density[x]))
    
    for i, order in enumerate(orders):
        if elapsed() > DEADLINE:
            break
        mask, count = greedy_construct(order)
        if count > best_count:
            best_count = count
            best_mask = mask
        mask, count = hill_climb(mask, count)
        if count > best_count:
            best_count = count
            best_mask = mask
    
    while elapsed() < DEADLINE - 5:
        perturbed_mask = best_mask
        perturbed_count = best_count
        num_removals = random.randint(3, 10)
        elements_in_best = [x for x in range(1, N+1) if best_mask & (1 << x)]
        if len(elements_in_best) <= num_removals:
            break
        to_remove = random.sample(elements_in_best, num_removals)
        for x in to_remove:
            result = try_remove(perturbed_mask, x, perturbed_count)
            if result is not None:
                perturbed_mask, perturbed_count = result
        
        mask, count = greedy_construct(list(range(1, N+1)), perturbed_mask)
        mask, count = hill_climb(mask, count)
        if count > best_count:
            best_count = count
            best_mask = mask
    
    result = [x for x in range(1, N+1) if best_mask & (1 << x)]
    if not result:
        result = [1, 2, 4, 8, 16, 32, 64, 128]
    return result
