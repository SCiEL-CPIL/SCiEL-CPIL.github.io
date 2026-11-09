import random
import time

def _check_divisibility(sum_set):
    """Check that no two distinct values in sum_set divide each other."""
    sums = sorted(sum_set)
    n = len(sums)
    for i in range(n):
        si = sums[i]
        for j in range(i + 1, n):
            sj = sums[j]
            if sj % si == 0:
                return False
    return True

def _get_subset_sums(elements):
    """Get all non-empty subset sums as a set."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _can_add(elements, x, current_sums):
    """Check if x can be added to elements without violating divisibility."""
    # New sums are: current_sums ∪ {s + x for s in current_sums} ∪ {x}
    
    # 1. Check x against current_sums
    for s in current_sums:
        if s == x:
            return False, None
        if x % s == 0:
            return False, None
        if s % x == 0:
            return False, None
            
    new_sums = set(current_sums)
    new_sums.add(x)
    
    # 2. Generate s+x and check against current_sums and other new sums
    generated = []
    for s in current_sums:
        ns = s + x
        generated.append(ns)
        # Check ns against x
        if ns == x: return False, None
        if ns % x == 0: return False, None
        if x % ns == 0: return False, None
        
        # Check ns against current_sums
        for t in current_sums:
            if t == ns:
                return False, None
            if ns > t:
                if ns % t == 0:
                    return False, None
            else:
                if t % ns == 0:
                    return False, None
                    
    # 3. Check generated sums against each other
    for i in range(len(generated)):
        a = generated[i]
        for j in range(i + 1, len(generated)):
            b = generated[j]
            if a == b:
                return False, None
            if b > a:
                if b % a == 0:
                    return False, None
            else:
                if a % b == 0:
                    return False, None
                    
    new_sums.update(generated)
    return True, new_sums

def _greedy_construct(pool, deadline, rng):
    """Greedy construction: add elements one by one if they don't violate constraints."""
    elements = []
    current_sums = set()
    
    for x in pool:
        if time.time() > deadline:
            break
        ok, new_sums = _can_add(elements, x, current_sums)
        if ok:
            elements.append(x)
            current_sums = new_sums
            
    return elements

def _local_search(elements, pool, deadline, rng):
    """Try to improve the set by swapping elements."""
    current_elements = list(elements)
    current_sums = _get_subset_sums(current_elements)
    best_size = len(current_elements)
    best_set = list(current_elements)
    
    # Create a set of available numbers
    available = [x for x in pool if x not in current_elements]
    
    while time.time() < deadline:
        if len(current_elements) == 0:
            break
            
        # Try to add a random available number
        if available:
            idx = rng.randint(0, len(available) - 1)
            x = available[idx]
            ok, new_sums = _can_add(current_elements, x, current_sums)
            if ok:
                current_elements.append(x)
                current_sums = new_sums
                available.remove(x)
                if len(current_elements) > best_size:
                    best_size = len(current_elements)
                    best_set = list(current_elements)
                    
        # Try to swap: remove one, add another
        if current_elements:
            remove_idx = rng.randint(0, len(current_elements) - 1)
            removed = current_elements.pop(remove_idx)
            # Recalculate sums
            current_sums = _get_subset_sums(current_elements)
            
            # Try to add a different number
            if available:
                # Pick a random available
                add_idx = rng.randint(0, len(available) - 1)
                add_x = available[add_idx]
                ok, new_sums = _can_add(current_elements, add_x, current_sums)
                if ok:
                    current_elements.append(add_x)
                    current_sums = new_sums
                    available.remove(add_x)
                    available.append(removed)
                    if len(current_elements) > best_size:
                        best_size = len(current_elements)
                        best_set = list(current_elements)
                else:
                    # Revert
                    current_elements.append(removed)
                    current_sums = _get_subset_sums(current_elements)
                    # add_x was not added, so available didn't change (we didn't remove it)
                    # Wait, we didn't remove add_x from available in the 'ok' branch unless it succeeded.
                    # In the 'not ok' branch, we should ensure add_x is still in available.
                    # It is, because we only removed it in the 'ok' branch.
                    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_result = [31, 32, 34]
    rng = random.Random(42)
    
    # Pools
    pool_large = list(range(128, 257))
    pool_medium = list(range(100, 257))
    pool_all = list(range(1, 257))
    
    # 1. Greedy Random Restarts
    for trial in range(1000):
        if time.time() > deadline - 5:
            break
            
        if trial % 3 == 0:
            pool = pool_large
        elif trial % 3 == 1:
            pool = pool_medium
        else:
            pool = pool_all
            
        pool_copy = pool[:]
        rng.shuffle(pool_copy)
        
        result = _greedy_construct(pool_copy, deadline, rng)
        
        if len(result) > len(best_result):
            best_result = result
            
    # 2. Local Search on Best Result
    if len(best_result) < 30 and time.time() < deadline - 2:
        # Use a broader pool for local search
        search_pool = list(range(1, 257))
        best_result = _local_search(best_result, search_pool, deadline, rng)
        
    # 3. Final Extension Attempt
    if time.time() < deadline:
        current_elements = list(best_result)
        current_sums = _get_subset_sums(current_elements)
        for x in range(1, 257):
            if x in current_elements:
                continue
            if time.time() > deadline:
                break
            ok, new_sums = _can_add(current_elements, x, current_sums)
            if ok:
                current_elements.append(x)
                current_sums = new_sums
                if len(current_elements) > len(best_result):
                    best_result = current_elements

    # Ensure return type is correct
    result = [int(x) for x in best_result]
    if len(set(result)) != len(result):
        result = list(set(result))
    result = [x for x in result if 1 <= x <= 256]
    result = sorted(result)
    
    # Final validation
    if len(result) > 0:
        sums = _get_subset_sums(result)
        if not _check_divisibility(sums):
            result = [31, 32, 34]
            
    return result
