"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin from 90s limit
    
    best_set = []
    best_len = 0
    
    def try_greedy(order):
        """Try to build a valid set by greedily adding elements in the given order."""
        selected = []
        n = 0
        
        # dp_all[k] is a bitmask where bit s is set if sum s is achievable with exactly k elements
        dp_all = [1]  # dp_all[0] has bit 0 set (empty subset, sum 0)
        # dp_without[a] is a list of bitmasks for sums achievable with k elements from selected excluding a
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= 48:
                break
            
            # Check 1: Is x the mean of any subset of selected of size >= 2?
            # Need: exists k >= 2, subset of selected with sum = k*x
            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target > 24576:  # max possible sum
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: For each a in selected, is a the mean of a subset involving x?
            # Need: exists a in selected, exists k >= 1, subset U of (selected \ {a}) with
            # sum(U) = (k+1)*a - x
            for a in selected:
                if not valid:
                    break
                # dp_without[a][k] gives achievable sums with k elements from selected\{a}
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > 24576:
                        continue
                    if (dp_without[a][k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add x to selected
            selected.append(x)
            
            # Update dp_all: new_dp_all[k] = dp_all[k] | (dp_all[k-1] << x)
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:  # all except x
                old_dp_w = dp_without[a]
                # new dp_w[k] = old_dp_w[k] | (old_dp_w[k-1] << x)
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For new element x, dp_without[x] is the old dp_all (selected before adding x)
            new_dp_without[x] = dp_all[:n+1] + [0] * max(0, n + 2 - (n + 1))
            
            dp_without = new_dp_without
            n += 1
        
        return selected
    
    def modular_order(mod):
        """Create ordering based on modular arithmetic - elements with same remainder mod 'mod' grouped."""
        buckets = [[] for _ in range(mod)]
        for x in range(1, 513):
            buckets[x % mod].append(x)
        order = []
        # Interleave buckets to spread out similar remainders
        max_len = max(len(b) for b in buckets) if buckets else 0
        for i in range(max_len):
            for b in buckets:
                if i < len(b):
                    order.append(b[i])
        return order
    
    def weighted_random_order(seed_val, weight_func):
        """Create a weighted random ordering."""
        random.seed(seed_val)
        candidates = list(range(1, 513))
        weights = [weight_func(x) for x in candidates]
        # Weighted shuffle
        order = []
        remaining = list(zip(candidates, weights))
        while remaining:
            total_w = sum(w for _, w in remaining)
            if total_w == 0:
                idx = random.randint(0, len(remaining) - 1)
            else:
                r = random.uniform(0, total_w)
                cumsum = 0
                idx = 0
                for i, (_, w) in enumerate(remaining):
                    cumsum += w
                    if r <= cumsum:
                        idx = i
                        break
            order.append(remaining[idx][0])
            remaining.pop(idx)
        return order
    
    # Strategy 1: Ascending order
    res = try_greedy(range(1, 513))
    if len(res) > best_len:
        best_set = res
        best_len = len(res)
    
    # Strategy 2: Descending order
    if time.time() - start_time < deadline - 5:
        res = try_greedy(range(512, 0, -1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
    
    # Strategy 3: Spread ordering (alternate low and high)
    if time.time() - start_time < deadline - 5:
        candidates = list(range(1, 513))
        spread = []
        lo, hi = 0, len(candidates) - 1
        while lo <= hi:
            spread.append(candidates[lo])
            if lo != hi:
                spread.append(candidates[hi])
            lo += 1
            hi -= 1
        res = try_greedy(spread)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
    
    # Strategy 4: Modular orderings with different moduli
    for mod in [7, 11, 13, 17, 19]:
        if time.time() - start_time > deadline - 3:
            break
        order = modular_order(mod)
        res = try_greedy(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
    
    # Strategy 5: Weighted random orders - prefer extremes
    for i in range(10):
        if time.time() - start_time > deadline - 3:
            break
        # Weight toward extremes (1 and 512)
        def extreme_weight(x):
            dist_from_center = abs(x - 256.5)
            return dist_from_center ** 2 + 1
        order = weighted_random_order(i * 7919 + 13, extreme_weight)
        res = try_greedy(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
    
    # Strategy 6: More pure random restarts
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 2:
        max_attempts = min(int(remaining_time * 2), 80)
        for i in range(max_attempts):
            if time.time() - start_time > deadline - 2:
                break
            random.seed(i * 31337 + 42)
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            res = try_greedy(candidates)
            if len(res) > best_len:
                best_set = res
                best_len = len(res)
                if best_len >= 48:
                    break
    
    # Strategy 7: Local search - try to improve by swapping rejected elements
    if best_len < 48 and time.time() - start_time < deadline - 5:
        # Try to extend the best set by trying to add more elements
        extended_order = [x for x in range(1, 513) if x not in best_set]
        # Sort by some heuristic
        extended_order.sort(key=lambda x: abs(x - 256))
        res = try_greedy(best_set + extended_order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
    
    return best_set

# EVOLVE-BLOCK-END
