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
    
    # Remove duplicates just in case
    selected = list(set(selected))
    n = len(selected)
    if n < 3:
        return True
    
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m < 2:
            continue
            
        # DP: dp[k] is a bitset where bit s is set if sum s is achievable with exactly k elements from 'others'
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in others:
            for k in range(m, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
                    
        # Check if any k in [2, m] has a target sum k*a present in dp[k]
        for k in range(2, m + 1):
            target = k * a
            if target <= 512 * m:  # bounds check
                if dp[k] & (1 << target):
                    return False
    return True

def _generate_23_candidates(max_val=512):
    """Generate all numbers of form 2^i * 3^j <= max_val."""
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

def _fast_validity_check_incr(existing_set, new_elem):
    """Fast check: can we add new_elem to existing_set while maintaining validity?
    Only checks:
    1. new_elem is not mean of subset of existing (size>=2)
    2. No existing elem is mean of subset including new_elem
    """
    existing = list(existing_set)
    n = len(existing)
    
    # Check 1: new_elem not mean of subset of existing
    # Need subset of existing, size k>=2, sum = k * new_elem
    if n >= 2:
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in existing:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        for k in range(2, n + 1):
            target = k * new_elem
            if target <= 512 * n:
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
            if target <= 512 * m:
                if dp[k] & (1 << target):
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    best_set = []
    
    # Generate multiplicative-structure candidates: 2^i * 3^j
    candidates_23 = _generate_23_candidates(512)
    
    # Also generate a broader pool: all numbers, but prioritize 2^i*3^j
    all_candidates = list(range(1, 513))
    
    # Strategy 1: Greedy over 2^i * 3^j candidates (sorted in various orders)
    orders = [
        candidates_23,
        sorted(candidates_23, reverse=True),
        list(candidates_23),
    ]
    # Shuffle some
    for _ in range(3):
        lst = candidates_23[:]
        random.shuffle(lst)
        orders.append(lst)
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        current_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if _fast_validity_check_incr(current, x):
                current.append(x)
                current_set.add(x)
                if len(current) > len(best_set):
                    best_set = current[:]
    
    # Strategy 2: Try to expand best_set with any remaining numbers
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        # Prioritize 2^i * 3^j in remaining
        remaining.sort(key=lambda x: (x not in set(candidates_23), x))
        random.shuffle([x for x in remaining if x in set(candidates_23)])
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in remaining[:]:
                if time.time() > deadline:
                    break
                if x in best_set_set:
                    continue
                if _fast_validity_check_incr(best_set, x):
                    best_set.append(x)
                    best_set_set.add(x)
                    remaining.remove(x)
                    improved = True
    
    # Strategy 3: Random restarts with structured candidates
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Mix of 2^i*3^j and random
        pool = candidates_23 + random.sample([x for x in range(1, 513) if x not in set(candidates_23)], min(50, len([x for x in range(1, 513) if x not in set(candidates_23)])))
        random.shuffle(pool)
        
        current = []
        current_set = set()
        for x in pool:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if _fast_validity_check_incr(current, x):
                current.append(x)
                current_set.add(x)
        
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Strategy 4: Local search - try swapping elements in best_set to allow more additions
    if best_set and time.time() < deadline:
        best_set_set = set(best_set)
        missing = [x for x in range(1, 513) if x not in best_set_set]
        
        # Try removing one element and adding two
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            temp_set = best_set[:i] + best_set[i+1:]
            temp_set_set = set(temp_set)
            
            # Try to add two elements
            added = []
            for x in missing:
                if time.time() > deadline:
                    break
                if x in temp_set_set or x in added:
                    continue
                if _fast_validity_check_incr(temp_set, x):
                    temp_set.append(x)
                    temp_set_set.add(x)
                    added.append(x)
                    if len(added) == 2:
                        break
            
            if len(added) >= 2 and len(temp_set) > len(best_set):
                best_set = temp_set
                best_set_set = set(best_set)
                missing = [x for x in range(1, 513) if x not in best_set_set]
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
