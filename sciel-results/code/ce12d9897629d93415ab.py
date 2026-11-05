"""Combined solver: multiplicative-structure candidates + remove-one-add-two local search."""
import time
import random

def _valid_mean_free(selected):
    """Full verification: no element is the mean of any subset of >=2 OTHER elements."""
    n = len(selected)
    if n < 3:
        return True
    selected = list(set(selected))
    n = len(selected)
    if n < 3:
        return True
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m < 2:
            continue
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        for k in range(2, m + 1):
            target = k * a
            if dp[k] & (1 << target):
                return False
    return True

def _generate_23_candidates(max_val=512):
    """All numbers of form 2^i * 3^j <= max_val."""
    candidates = set()
    i = 0
    while (1 << i) <= max_val:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= max_val:
            candidates.add(val_i * (3 ** j))
            j += 1
        i += 1
    return sorted(candidates)

def _generate_pools(max_val=512):
    """Generate diverse candidate pools."""
    pool_23 = _generate_23_candidates(max_val)
    
    pool_pow2 = set()
    i = 0
    while (1 << i) <= max_val:
        pool_pow2.add(1 << i)
        i += 1
    
    pool_exp3 = set()
    val = 1
    while val <= max_val:
        pool_exp3.add(val)
        val *= 3
    
    pool_exp4 = set()
    val = 1
    while val <= max_val:
        pool_exp4.add(val)
        val *= 4
    
    pool_fib = set()
    a, b = 1, 2
    while a <= max_val:
        pool_fib.add(a)
        a, b = b, a + b
    
    all_candidates = list(range(1, max_val + 1))
    
    return {
        '23': pool_23,
        'pow2': sorted(pool_pow2),
        'exp3': sorted(pool_exp3),
        'exp4': sorted(pool_exp4),
        'fib': sorted(pool_fib),
        'all': all_candidates,
        '23_set': set(pool_23),
    }

def _fast_add_check(existing, new_elem):
    """Check if new_elem can be added to existing while maintaining mean-free property.
    Only checks new constraints introduced by adding new_elem."""
    n = len(existing)
    
    # Check 1: new_elem is not mean of any subset of existing (size>=2)
    if n >= 2:
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in existing:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        for k in range(2, n + 1):
            target = k * new_elem
            if dp[k] & (1 << target):
                return False
    
    # Check 2: For each existing elem a, no subset of (existing \ {a}) + {new_elem} of size>=2 averages to a
    for i, a in enumerate(existing):
        others = [x for j, x in enumerate(existing) if j != i]
        others.append(new_elem)
        m = len(others)
        if m < 2:
            continue
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        for k in range(2, m + 1):
            target = k * a
            if dp[k] & (1 << target):
                return False
    
    return True

def _greedy_construct(pool, deadline):
    """Greedy construction from a candidate pool."""
    current = []
    current_set = set()
    for x in pool:
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if _fast_add_check(current, x):
            current.append(x)
            current_set.add(x)
    return current

def _expand_set(best_set, deadline, priority_set=None):
    """Expand best_set by trying to add any remaining numbers, prioritizing 2^i*3^j."""
    if not best_set:
        return best_set
    best_set = best_set[:]
    best_set_set = set(best_set)
    
    if priority_set is None:
        priority_set = set()
    
    remaining = [x for x in range(1, 513) if x not in best_set_set]
    # Sort: prioritize 2^i*3^j numbers first, then by value
    remaining.sort(key=lambda x: (0 if x in priority_set else 1, x))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in remaining[:]:
            if time.time() > deadline:
                break
            if x in best_set_set:
                continue
            if _fast_add_check(best_set, x):
                best_set.append(x)
                best_set_set.add(x)
                remaining.remove(x)
                improved = True
    return best_set

def _local_search_remove_one_add_two(best_set, deadline, priority_set=None):
    """Local search: try removing one element and adding two others to grow the set."""
    if not best_set or len(best_set) < 3:
        return best_set
    
    best_set = best_set[:]
    best_set_set = set(best_set)
    missing = [x for x in range(1, 513) if x not in best_set_set]
    
    if priority_set is None:
        priority_set = set()
    
    # Sort missing to prioritize 2^i*3^j
    missing.sort(key=lambda x: (0 if x in priority_set else 1, x))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            temp_set = best_set[:i] + best_set[i+1:]
            temp_set_set = set(temp_set)
            
            # Try to add as many elements as possible from missing
            added = []
            for x in missing:
                if time.time() > deadline:
                    break
                if x in temp_set_set or x in added:
                    continue
                if _fast_add_check(temp_set, x):
                    temp_set.append(x)
                    temp_set_set.add(x)
                    added.append(x)
                    if len(added) >= 2:
                        break
            
            # If we added at least 2, we improved (net gain of 1+)
            if len(added) >= 2 and len(temp_set) > len(best_set):
                best_set = temp_set
                best_set_set = set(best_set)
                missing = [x for x in range(1, 513) if x not in best_set_set]
                missing.sort(key=lambda x: (0 if x in priority_set else 1, x))
                improved = True
    
    return best_set

def _local_search_remove_one_add_one(best_set, deadline, priority_set=None):
    """Local search: try removing one element and adding one other (swap)."""
    if not best_set or len(best_set) < 3:
        return best_set
    
    best_set = best_set[:]
    best_set_set = set(best_set)
    missing = [x for x in range(1, 513) if x not in best_set_set]
    
    if priority_set is None:
        priority_set = set()
    
    missing.sort(key=lambda x: (0 if x in priority_set else 1, x))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            temp_set = best_set[:i] + best_set[i+1:]
            temp_set_set = set(temp_set)
            
            # Try to add one element
            for x in missing:
                if time.time() > deadline:
                    break
                if x in temp_set_set:
                    continue
                if _fast_add_check(temp_set, x):
                    temp_set.append(x)
                    temp_set_set.add(x)
                    # After swap, try to expand
                    temp_set = _expand_set(temp_set, deadline, priority_set)
                    if len(temp_set) > len(best_set):
                        best_set = temp_set
                        best_set_set = set(best_set)
                        missing = [x for x in range(1, 513) if x not in best_set_set]
                        missing.sort(key=lambda x: (0 if x in priority_set else 1, x))
                        improved = True
                    break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    best_set = []
    
    pools = _generate_pools(512)
    pool_23 = pools['23']
    pool_23_set = pools['23_set']
    
    # Strategy 1: Greedy over various structured pools
    orders = [
        pool_23,
        sorted(pool_23, reverse=True),
        pools['pow2'],
        pools['exp3'],
        pools['exp4'],
        pools['fib'],
    ]
    
    # Add randomized orders of 2^i*3^j
    for _ in range(5):
        lst = pool_23[:]
        random.shuffle(lst)
        orders.append(lst)
    
    for order in orders:
        if time.time() > deadline:
            break
        current = _greedy_construct(order, deadline)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Strategy 2: Expand best set with priority on 2^i*3^j
    if best_set:
        best_set = _expand_set(best_set, deadline, pool_23_set)
    
    # Strategy 3: Random restarts with mixed pools
    for iteration in range(40):
        if time.time() > deadline:
            break
        
        # Mix of structured and random
        structured_choice = random.choice([pool_23, pools['pow2'], pools['exp3'], pools['fib']])
        non_23 = [x for x in range(1, 513) if x not in pool_23_set]
        random_part = random.sample(non_23, min(30, len(non_23)))
        pool = structured_choice + random_part
        random.shuffle(pool)
        
        current = _greedy_construct(pool, deadline)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Strategy 4: Local search - remove one, add two (most impactful)
    if best_set and time.time() < deadline:
        best_set = _local_search_remove_one_add_two(best_set, deadline, pool_23_set)
    
    # Strategy 5: Expand after local search
    if best_set and time.time() < deadline:
        best_set = _expand_set(best_set, deadline, pool_23_set)
    
    # Strategy 6: Local search - remove one, add one (fine-tuning)
    if best_set and time.time() < deadline:
        best_set = _local_search_remove_one_add_one(best_set, deadline, pool_23_set)
    
    # Final expansion
    if best_set and time.time() < deadline:
        best_set = _expand_set(best_set, deadline, pool_23_set)
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
