"""Search-based solver using multiplicative-structure candidates for mean-free subset problem."""
import time
import random

def _valid_mean_free(selected):
    """Returns True if the list 'selected' satisfies the condition that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    Uses bitset DP for speed."""
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

def _generate_candidates(max_val=512):
    """Generate diverse candidate pools."""
    # Pool 1: 2^i * 3^j
    pool_23 = set()
    i = 0
    while (1 << i) <= max_val:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= max_val:
            pool_23.add(val_i * (3 ** j))
            j += 1
        i += 1
    
    # Pool 2: Powers of 2
    pool_pow2 = set()
    i = 0
    while (1 << i) <= max_val:
        pool_pow2.add(1 << i)
        i += 1
    
    # Pool 3: Exponential spacing (base 3)
    pool_exp3 = set()
    val = 1
    while val <= max_val:
        pool_exp3.add(val)
        val *= 3
    
    # Pool 4: Exponential spacing (base 4)
    pool_exp4 = set()
    val = 1
    while val <= max_val:
        pool_exp4.add(val)
        val *= 4
    
    # Pool 5: Fibonacci-like
    pool_fib = set()
    a, b = 1, 2
    while a <= max_val:
        pool_fib.add(a)
        a, b = b, a + b
    
    # Combined pool with priority
    all_candidates = set(range(1, max_val + 1))
    
    return {
        '23': sorted(pool_23),
        'pow2': sorted(pool_pow2),
        'exp3': sorted(pool_exp3),
        'exp4': sorted(pool_exp4),
        'fib': sorted(pool_fib),
        'all': sorted(all_candidates),
    }

def _fast_add_check(existing, new_elem):
    """Fast check: can we add new_elem to existing while maintaining validity?
    Only checks new constraints:
    1. new_elem is not mean of subset of existing (size>=2)
    2. No existing elem is mean of subset including new_elem
    """
    n = len(existing)
    
    # Check 1: new_elem not mean of subset of existing
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

def _expand_set(best_set, deadline):
    """Try to expand best_set by adding any remaining numbers."""
    if not best_set:
        return best_set
    best_set = best_set[:]
    best_set_set = set(best_set)
    remaining = [x for x in range(1, 513) if x not in best_set_set]
    
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

def _local_search_swap(best_set, deadline):
    """Local search: try swapping elements to allow more additions."""
    if not best_set or len(best_set) < 3:
        return best_set
    
    best_set = best_set[:]
    best_set_set = set(best_set)
    missing = [x for x in range(1, 513) if x not in best_set_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            for x in missing[:]:
                if time.time() > deadline:
                    break
                # Try swapping best_set[i] with x
                old = best_set[i]
                temp_set = best_set[:i] + best_set[i+1:]
                temp_set.append(x)
                temp_set_set = set(temp_set)
                
                if _fast_add_check(temp_set, old):
                    # Check if old can be added back
                    if _fast_add_check(temp_set, old):
                        # This swap doesn't help, skip
                        continue
                
                # Try to expand the swapped set
                expanded = _expand_set(temp_set, deadline)
                if len(expanded) > len(best_set):
                    best_set = expanded
                    best_set_set = set(best_set)
                    missing = [x for x in range(1, 513) if x not in best_set_set]
                    improved = True
                    break
            if improved:
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 40.0  # Safety margin before 60s limit
    best_set = []
    
    candidates = _generate_candidates(512)
    
    # Strategy 1: Greedy over various structured pools
    pools_to_try = [
        candidates['23'],
        sorted(candidates['23'], reverse=True),
        candidates['pow2'],
        candidates['exp3'],
        candidates['exp4'],
        candidates['fib'],
    ]
    
    # Add some randomized orders
    for _ in range(5):
        pool = candidates['23'][:]
        random.shuffle(pool)
        pools_to_try.append(pool)
    
    for pool in pools_to_try:
        if time.time() > deadline:
            break
        current = _greedy_construct(pool, deadline)
        if len(current) > len(best_set):
            best_set = current
    
    # Strategy 2: Expand best set
    if best_set:
        best_set = _expand_set(best_set, deadline)
    
    # Strategy 3: Random restarts with mixed pools
    for iteration in range(30):
        if time.time() > deadline:
            break
        
        # Mix of structured and random
        structured = random.choice([candidates['23'], candidates['pow2'], candidates['exp3'], candidates['fib']])
        random_part = random.sample(candidates['all'], min(30, len(candidates['all'])))
        pool = structured + random_part
        random.shuffle(pool)
        
        current = _greedy_construct(pool, deadline)
        if len(current) > len(best_set):
            best_set = current
    
    # Strategy 4: Local search
    if best_set and time.time() < deadline:
        best_set = _local_search_swap(best_set, deadline)
    
    # Final expansion
    if best_set and time.time() < deadline:
        best_set = _expand_set(best_set, deadline)
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
