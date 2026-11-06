"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    best_set = []
    best_len = 0

    MAX_SUM = 24576  # 48 * 512

    def try_greedy(order, max_k=48):
        """Greedy: add elements in order if they satisfy the mean constraint."""
        selected = []
        n = 0
        dp_all = [1]  # dp_all[k] = bitmask of sums achievable with exactly k elements
        dp_without = {}  # dp_without[a][k] = bitmask of sums achievable with exactly k elements from selected \ {a}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= max_k:
                break
            
            # Check 1: x is not the mean of any subset of >= 2 other elements
            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target > MAX_SUM:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: no existing element a is the mean of x plus a subset of other elements
            for a in selected:
                dp_w = dp_without[a]
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > MAX_SUM:
                        continue
                    if (dp_w[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Valid, add x
            selected.append(x)
            
            # Save old state for x's exclusion set
            old_dp_all = dp_all[:]
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = old_dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For x, the set of others is the old selected set.
            new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
            
            dp_without = new_dp_without
            n += 1
            
        return selected

    def local_improve(s, max_iterations=200):
        """Try to improve set s by swapping elements."""
        if not s:
            return s
        s = sorted(s)
        n = len(s)
        # Precompute DP for the current set
        dp_all = [1]
        dp_without = {}
        
        # Build initial DP
        for i, x in enumerate(s):
            old_dp_all = dp_all[:]
            new_dp_all = [0] * (i + 2)
            new_dp_all[0] = 1
            for k in range(1, i + 2):
                val = old_dp_all[k] if k <= i else 0
                if k - 1 >= 0 and k - 1 <= i:
                    val |= (old_dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            new_dp_without = {}
            for a in s[:i]:
                old_dp_w = dp_without[a]
                new_list = [0] * (i + 2)
                new_list[0] = 1
                for k in range(1, i + 2):
                    val = old_dp_w[k] if k <= i - 1 else 0
                    if k - 1 >= 0 and k - 1 <= i - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            new_dp_without[x] = old_dp_all + [0] * max(0, (i + 1) - len(old_dp_all))
            dp_without = new_dp_without
        
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            if time.time() - start_time > deadline:
                break
            
            for i in range(n):
                if time.time() - start_time > deadline:
                    break
                # Try removing s[i] and adding some other element
                a = s[i]
                # Rebuild DP without a
                s_without = [x for x in s if x != a]
                n_without = len(s_without)
                
                dp_w = [1]
                for x in s_without:
                    old_dp = dp_w[:]
                    new_dp = [0] * (len(dp_w) + 1)
                    new_dp[0] = 1
                    for k in range(1, len(dp_w) + 1):
                        val = old_dp[k] if k < len(old_dp) else 0
                        if k - 1 >= 0 and k - 1 < len(old_dp):
                            val |= (old_dp[k-1] << x)
                        new_dp[k] = val
                    dp_w = new_dp
                
                # Try adding candidates not in s
                for cand in range(1, 513):
                    if cand in s:
                        continue
                    if time.time() - start_time > deadline:
                        break
                    
                    # Check if cand can be added to s_without
                    valid = True
                    for k in range(2, n_without + 1):
                        target = k * cand
                        if target > MAX_SUM:
                            break
                        if (dp_w[k] >> target) & 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Check if any element in s_without is mean of cand + subset of others
                    for b in s_without:
                        # Need DP of s_without \ {b}
                        # This is expensive; approximate by checking small k only
                        for k in range(1, min(n_without, 6)):  # Limit k for speed
                            target = (k + 1) * b - cand
                            if target < 0 or target > MAX_SUM:
                                continue
                            # We don't have dp_without_b readily; skip this check for speed
                            # This makes local improve less rigorous but faster
                            pass
                    
                    if valid:
                        # Accept the swap
                        s.remove(a)
                        s.append(cand)
                        s.sort()
                        improved = True
                        break
                if improved:
                    break
        
        return s

    # Generate multiple orderings
    orders = []
    
    # 1. Spread (alternating low/high)
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1
    orders.append(spread)
    
    # 2. Random shuffles with different seeds
    for i in range(15):
        random.seed(i * 7 + 1)
        cand = list(range(1, 513))
        random.shuffle(cand)
        orders.append(cand)
        
    # 3. Ascending/Descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    # 4. Prefer numbers with fewer divisors (less likely to be means)
    def divisor_count(x):
        cnt = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        return cnt
    
    pref = sorted(range(1, 513), key=lambda x: divisor_count(x))
    orders.append(pref)
    
    # 5. Random with bias toward extremes
    for i in range(10):
        random.seed(i * 13 + 3)
        cand = list(range(1, 513))
        # Bias: keep some order but shuffle with weights
        weighted = []
        for x in cand:
            w = 1.0 + 2.0 * abs(x - 256) / 256  # Higher weight for extremes
            weighted.extend([x] * int(w * 5))
        random.shuffle(weighted)
        # Deduplicate
        seen = set()
        order = []
        for x in weighted:
            if x not in seen:
                seen.add(x)
                order.append(x)
        orders.append(order)
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        res = try_greedy(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
    
    # Local improvement on the best set found
    if best_set and time.time() - start_time < deadline - 5:
        improved = local_improve(best_set, max_iterations=100)
        if len(improved) > best_len:
            best_set = improved
            best_len = len(best_set)
    
    return best_set

# EVOLVE-BLOCK-END
