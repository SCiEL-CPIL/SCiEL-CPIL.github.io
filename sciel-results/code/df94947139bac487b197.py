import time
import random

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def check_new_element(a, existing):
        """Check if a can be added to existing set without violating the mean condition.
        Only need to check: is a the mean of any subset of >=2 elements from existing?
        Also need to check: for each element b in existing, is b the mean of any subset 
        of >=2 elements from (existing - {b}) + {a}?
        But the second check is expensive. Let's first do the easy check."""
        m = len(existing)
        if m < 2:
            return True
        
        # Check 1: Is a the mean of any subset of >=2 elements from existing?
        # Need: sum(subset) = k * a for some k >= 2, subset size = k
        # So we need to check if k*a is achievable as sum of exactly k elements from existing
        total = sum(existing)
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in existing:
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        
        for k in range(2, m + 1):
            target = k * a
            if target > total:
                break
            if (dp[k] >> target) & 1:
                return False
        
        # Check 2: For each b in existing, is b the mean of any subset of >=2 
        # elements from (existing - {b}) + {a}?
        # This is more complex. We'll do a lighter check: 
        # For the new element a to cause a violation, there must exist a subset T 
        # of (existing - {b}) + {a} with |T| >= 2 and sum(T) = |T| * b.
        # T either contains a or not.
        # If T doesn't contain a, it's already checked (existing was valid).
        # If T contains a, then T = {a} ∪ U where U ⊆ (existing - {b}), |U| >= 1,
        # and a + sum(U) = (|U|+1) * b
        # => sum(U) = (|U|+1)*b - a
        # For each b, we need to check if there exists U ⊆ (existing-{b}), |U|>=1,
        # such that sum(U) = (|U|+1)*b - a.
        
        # This is still expensive for all b. Let's do a heuristic: only check if 
        # the violation is "close" - i.e., for small subsets.
        # Actually, let's just do the full check but optimize.
        
        for b in existing:
            others = [x for x in existing if x != b]
            # We need to check subsets of others + {a} that include a
            # T = {a} ∪ U, U ⊆ others, |U| >= 1
            # sum(T) = a + sum(U) = (|U|+1)*b
            # sum(U) = (|U|+1)*b - a
            n_others = len(others)
            if n_others < 1:
                continue
            
            # DP over others to find achievable sums for each cardinality
            dp2 = [0] * (n_others + 1)
            dp2[0] = 1
            
            for x in others:
                for k in range(n_others - 1, -1, -1):
                    if dp2[k]:
                        dp2[k+1] |= (dp2[k] << x)
            
            # Check for each j >= 1: is (j+1)*b - a achievable as sum of exactly j elements?
            max_sum = sum(others)
            for j in range(1, n_others + 1):
                target = (j + 1) * b - a
                if target < 0:
                    continue
                if target > max_sum:
                    break
                if (dp2[j] >> target) & 1:
                    return False
        
        return True

    def try_greedy(candidates):
        current = []
        for x in candidates:
            if len(current) >= 48:
                break
            if time.time() > deadline:
                break
            if check_new_element(x, current):
                current.append(x)
        return current

    def local_search(S):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= 48:
                    break
                if check_new_element(x, S):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= 48:
                break
            
            # Try swapping: remove one element, try to add a different one
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = S_set - {removed}
                    for x in available:
                        if x in S_temp_set:
                            continue
                        if check_new_element(x, S_temp):
                            S = S_temp + [x]
                            S_set = S_temp_set | {x}
                            available.remove(x)
                            available.append(removed)
                            improved = True
                            break
                    if improved:
                        break
        return S

    # 1. Structured Greedy Passes
    # Descending
    desc = list(range(512, 0, -1))
    res = try_greedy(desc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Ascending
    asc = list(range(1, 513))
    res = try_greedy(asc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Random Passes
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        res = try_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
        if best_size == 48:
            return best_set

    # 2. Local Search
    if best_set and time.time() < deadline:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == 48:
            return best_set

    # 3. More Local Search with restarts
    if best_set and time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            perturbed = list(best_set)
            rng = random.Random(time.time_ns())
            if perturbed:
                # Remove 1-2 random elements
                n_remove = min(rng.randint(1, 2), len(perturbed))
                for _ in range(n_remove):
                    if perturbed:
                        idx = rng.randrange(len(perturbed))
                        perturbed.pop(idx)
            improved = local_search(perturbed)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == 48:
                return best_set

    return best_set
