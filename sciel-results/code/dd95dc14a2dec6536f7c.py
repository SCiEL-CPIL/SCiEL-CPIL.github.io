import time
import random

def construct():
    """
    Construct a set of up to 48 distinct positive integers <= 512 such that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    best_set = []

    # Strategy 1: Greedy Descending with pre-computed DP
    if time.time() < deadline:
        s = _greedy_construct(range(512, 0, -1), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set

    # Strategy 2: Greedy Ascending
    if time.time() < deadline:
        s = _greedy_construct(range(1, 513), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set

    # Strategy 3: Random shuffles
    if len(best_set) < 48 and time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            s = _greedy_construct(candidates, deadline)
            if len(s) > len(best_set):
                best_set = s
                if len(best_set) == 48:
                    break

    # Strategy 4: Local improvement
    if len(best_set) < 48 and time.time() < deadline:
        s = _local_improvement(best_set, deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set

    return best_set

def _greedy_construct(candidates, deadline):
    """
    Greedily add elements from candidates if valid, maintaining a global DP.
    """
    selected = []
    selected_set = set()
    
    # Global DP: dp[k] is bitset of achievable sums with k elements from selected
    max_n = 48
    dp = [0] * (max_n + 1)
    dp[0] = 1  # sum 0 with 0 elements

    for x in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if x in selected_set:
            continue
        
        # Check validity using global DP
        if _is_valid_addition_global(selected, x, dp):
            selected.append(x)
            selected_set.add(x)
            # Update global DP
            _update_dp(dp, x, len(selected) - 1)
            
    return selected

def _update_dp(dp, x, current_n):
    """Update the global DP by adding element x. current_n is the size BEFORE adding x."""
    for k in range(current_n, -1, -1):
        if dp[k]:
            dp[k + 1] |= (dp[k] << x)

def _is_valid_addition_global(selected, x, dp):
    """
    Check if adding x to selected maintains the no-mean property.
    Uses the precomputed dp for efficiency.
    1. x is not the mean of any subset of selected (size >= 2).
    2. No a in selected is the mean of any subset of selected + {x} (size >= 2) that includes x.
    """
    n = len(selected)

    # Check 1: Is x the mean of a subset of selected?
    # Sum of k elements = k * x, k >= 2.
    if n >= 2:
        max_sum = sum(selected)
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return False

    # Check 2: For each a in selected, is a the mean of a subset including x?
    # Subset T from selected \ {a}, size k >= 1.
    # Sum(T) + x = (k+1) * a
    # Sum(T) = (k+1) * a - x
    # We need to check if Sum(T) is achievable by k elements from selected \ {a}.

    for a in selected:
        # We need to check if there exists k >= 1 such that
        # sum of k elements from selected \ {a} equals (k+1)*a - x.
        
        # Optimization: If we can quickly rule out using modular arithmetic or bounds, do so.
        # But for now, let's use a targeted DP or check against the global DP with exclusion.
        
        # A faster way: 
        # If target = (k+1)*a - x is achievable by k elements from selected,
        # does it necessarily use a?
        # If target > sum(selected) - a, it must use a? No, that's not right.
        # If target is achievable by k elements from selected, and we want to know if it's achievable
        # WITHOUT a, we can check if target is achievable by k elements from selected \ {a}.
        
        # Efficient check:
        # Let S be selected.
        # We want to know if exists k>=1: sum_{y in T, y in S\{a}} y = (k+1)a - x, |T|=k.
        
        # Let's iterate k.
        # Max k is n-1.
        # Min target is 1*a - x. Max target is n*a - x.
        
        # We can build a small DP for S\{a} if n is small, but n can be up to 48.
        # However, we can use the global DP and "subtract" a's contribution if possible.
        # Specifically, if dp[k] has bit target set, it means there is a subset of size k summing to target.
        # This subset might contain a.
        # If it contains a, then sum(T') = target - a for T' of size k-1 from S\{a}.
        # So target is achievable from S\{a} with size k IF:
        # 1. (dp[k] >> target) & 1 is true AND the subset does NOT contain a.
        # OR
        # 2. There is a subset of size k from S\{a} summing to target.
        
        # A robust way without building full DP for S\{a} every time:
        # We can check if target is achievable from S\{a} with size k.
        # This is equivalent to:
        # (dp_excl_a[k] >> target) & 1
        # where dp_excl_a is DP for S\{a}.
        
        # Building dp_excl_a is O(n^2) per a, which is O(n^3) total. Too slow for n=48?
        # 48^3 = 110k operations. Actually, the bitset shift is expensive.
        # Let's try a different angle.
        
        # If n is small (< 20), we can afford to build the exclusion DP.
        # If n is large, we rely on the fact that if a is in the set, and we're checking many x's,
        # maybe we can cache? No, x changes.
        
        # Let's implement the exclusion DP check but optimize it.
        # We only need to check if ANY k works.
        
        if _is_mean_with_exclusion(selected, a, x):
            return False

    return True

def _is_mean_with_exclusion(selected, a, x):
    """
    Check if a is the mean of a subset of selected + {x} that includes x.
    Equivalent to: exists k >= 1 such that sum of k elements from selected \ {a} is (k+1)*a - x.
    """
    n = len(selected)
    others = [y for y in selected if y != a]
    if not others:
        return False
    
    m = len(others)
    max_sum_others = sum(others)
    
    # Build DP for others
    # dp[k] bitset
    dp = [0] * (m + 1)
    dp[0] = 1
    
    for elem in others:
        for k in range(m - 1, -1, -1):
            if dp[k]:
                dp[k + 1] |= (dp[k] << elem)
                
    for k in range(1, m + 1):
        target = (k + 1) * a - x
        if target < 0 or target > max_sum_others:
            continue
        if (dp[k] >> target) & 1:
            return True
            
    return False

def _local_improvement(current_set, deadline):
    """
    Try to improve the current set by swapping or adding elements.
    """
    if not current_set:
        return current_set
    
    current_list = list(current_set)
    current_set_copy = set(current_list)
    
    # Try adding elements not in the set
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(current_list) >= 48:
            break
        if x in current_set_copy:
            continue
        
        # Use a local check (slower but independent)
        if _is_valid_addition_local(current_list, x):
            current_list.append(x)
            current_set_copy.add(x)
    
    # Try swapping: remove one element, try to add one or two
    # This is expensive, so only do a few trials
    for i in range(len(current_list)):
        if time.time() > deadline:
            break
        if len(current_list) >= 48:
            break
            
        removed = current_list[i]
        temp_list = [x for j, x in enumerate(current_list) if j != i]
        temp_set = set(temp_list)
        
        # Try to add back a different element
        improved = False
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in temp_set or x == removed:
                continue
            if _is_valid_addition_local(temp_list, x):
                new_list = temp_list + [x]
                # Check if this new list is "better" or just valid?
                # We just want to maximize size, so if we can add, we do.
                # But wait, we removed one and added one, size is same.
                # This doesn't help increase size.
                # We need to remove one and add TWO to increase size by 1.
                # Or remove one and add one that allows future additions?
                # Let's stick to just adding for now in local improvement.
                pass
                
    return current_list

def _is_valid_addition_local(selected, x):
    """
    Slower, independent check for validity.
    """
    n = len(selected)
    
    # Check 1: x is not mean of subset of selected
    if n >= 2:
        max_sum = sum(selected)
        dp = [0] * (n + 1)
        dp[0] = 1
        for elem in selected:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= (dp[k] << elem)
        
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return False

    # Check 2: a is not mean of subset including x
    for a in selected:
        if _is_mean_with_exclusion(selected, a, x):
            return False

    return True
