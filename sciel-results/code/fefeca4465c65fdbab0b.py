import time
import random

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _ap_solution(k):
    """Return AP of size k if it fits in 1..256 and is valid, else None."""
    if k == 1:
        return [1]
    max_j_val = max(j * (k - 1 - j) for j in range(1, k))
    required_x_1 = max_j_val + 1
    if required_x_1 <= 256 - k + 1:
        x_1 = required_x_1
        set_vals = list(range(x_1, x_1 + k))
        if len(set(set_vals)) == k and all(1 <= v <= 256 for v in set_vals):
            if _is_valid_bitset(set_vals):
                return set_vals
    return None

def _greedy_construct(deadline):
    """Greedy construction with incremental bitset DP."""
    elements = []
    dp = [1]  # dp[k] = bitset of achievable sums with exactly k elements
    best_elements = []
    
    for x in range(1, 257):
        if time.time() > deadline - 5.0:
            break
        # Try to add x
        new_dp = [dp[k] for k in range(len(dp))]
        while len(new_dp) <= len(elements) + 1:
            new_dp.append(0)
        
        for k in range(len(elements), -1, -1):
            new_dp[k + 1] |= (dp[k] << x)
        
        # Check cardinality separation
        valid = True
        n_k = len(new_dp)
        for i in range(n_k):
            for j in range(i + 1, n_k):
                if new_dp[i] & new_dp[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            elements.append(x)
            dp = new_dp
            best_elements = list(elements)
            if len(elements) >= 60:
                break
    
    return best_elements

def _hill_climb(initial_set, deadline):
    """Try to add elements to the set via hill climbing."""
    s = list(initial_set)
    s_set = set(s)
    best = list(s)
    
    # Build initial dp
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in s:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for x in range(1, 257):
            if x in s_set:
                continue
            # Try adding x
            new_dp = [dp[k] for k in range(len(dp))]
            while len(new_dp) <= len(s) + 1:
                new_dp.append(0)
            
            for k in range(len(s), -1, -1):
                new_dp[k + 1] |= (dp[k] << x)
            
            valid = True
            n_k = len(new_dp)
            for i in range(n_k):
                for j in range(i + 1, n_k):
                    if new_dp[i] & new_dp[j]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                s.append(x)
                s_set.add(x)
                dp = new_dp
                best = list(s)
                improved = True
                break
    
    return best

def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    best_set = []
    
    # Phase 1: Fast AP search
    for k in range(60, 0, -1):
        sol = _ap_solution(k)
        if sol is not None:
            best_set = list(sol)
            break
    
    # Phase 2: Greedy construction
    greedy_set = _greedy_construct(deadline)
    if len(greedy_set) > len(best_set):
        best_set = greedy_set
    
    # Phase 3: Hill climbing on the best set found
    if time.time() < deadline - 10.0:
        climbed = _hill_climb(best_set, deadline)
        if len(climbed) > len(best_set):
            best_set = climbed
    
    # Ensure we return a valid list of Python ints
    result = list(best_set)
    if not result:
        result = [1]
    
    return result
