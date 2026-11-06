import time
import random

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality.
    dp[k] is an integer where bit s is set if sum s is achievable by k elements.
    """
    n = len(elements)
    if n == 0:
        return [0] * 1
    max_sum = sum(elements)
    # We only need bits up to max_sum
    # dp[k]
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for e in elements:
        # Update in reverse
        for k in range(n, 0, -1):
            # We shift dp[k-1] by e
            # We should mask to keep bits reasonable? 
            # max_sum is small enough (48*512 ~ 24k bits)
            dp[k] |= (dp[k-1] << e)
    return dp

def _is_valid_mean_free_with_dp(A, x):
    """
    Check if adding x to A keeps the set mean-free.
    A is a list of distinct integers.
    """
    n = len(A)
    
    # 1. Check if x is the mean of any subset of A of size k >= 2
    # Build DP for A
    dp_A = _build_dp(A)
    for k in range(2, n + 1):
        target = k * x
        if target < 0:
            continue
        # Check if bit target is set in dp_A[k]
        if (dp_A[k] >> target) & 1:
            return False
            
    # 2. Check if any a in A is the mean of any subset of (A U {x}) \ {a} of size k >= 2
    # Such a subset must contain x (since subsets of A \ {a} were already valid)
    # Let S be the subset. S contains x.
    # S = T U {x}, T subset of A \ {a}.
    # |S| = k. |T| = k - 1.
    # Sum(S) = Sum(T) + x = k * a.
    # Sum(T) = k * a - x.
    # We need to check if there exists T subset of A \ {a} with |T| = k - 1 and Sum(T) = k * a - x.
    
    # We can compute DP for A \ {a} for each a in A.
    # To optimize, we can just iterate through a in A.
    for a in A:
        # Build DP for A \ {a}
        others = [v for v in A if v != a]
        if len(others) < 1:
            continue # k-1 >= 1 implies k >= 2, but if others is empty, no T exists
        
        dp_others = _build_dp(others)
        
        # k ranges from 2 to len(A) (since |S| <= len(A) as x is added to A of size n, total size n+1, S is subset of size n, so k <= n)
        # Wait, S is subset of (A U {x}) \ {a}. Size of (A U {x}) \ {a} is n.
        # So k can be at most n.
        # |T| = k - 1. Max |T| is n - 1.
        # dp_others has indices up to len(others) = n - 1.
        
        for k in range(2, n + 1):
            target_sum = k * a - x
            if target_sum < 0:
                continue
            j = k - 1
            if j > len(others):
                continue
            # Check if bit target_sum is set in dp_others[j]
            if (dp_others[j] >> target_sum) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best_set = []
    best_len = 0
    
    # Multiple restarts with different random orders
    # Also try some heuristic orders
    orders = []
    
    # 1. Random orders
    for _ in range(50):
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        orders.append(candidates)
        
    # 2. Spread orders (e.g., alternating low/high)
    # 3. Just in case, ascending/descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = []
        for x in order:
            if time.time() > deadline:
                break
            
            if _is_valid_mean_free_with_dp(current_set, x):
                current_set.append(x)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
        # Early exit if we hit the cap
        if best_len >= 48:
            break
            
    return best_set
