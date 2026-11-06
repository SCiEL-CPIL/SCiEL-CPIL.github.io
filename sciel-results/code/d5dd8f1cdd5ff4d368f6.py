"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    best_set = []
    best_len = 0
    MAX_N = 512
    MAX_SUM = 24576  # 48 * 512

    def try_greedy_conflict_density(seed=0):
        """Greedy construction using conflict density heuristic."""
        random.seed(seed)
        remaining = set(range(1, MAX_N + 1))
        selected = []
        n = 0
        dp_all = [1]  # dp_all[k] is bitmask of achievable sums with exactly k elements
        dp_without = {}  # dp_without[a][k] is bitmask of sums achievable from selected \ {a} with exactly k elements
        
        while remaining and n < 48:
            if time.time() - start_time > deadline:
                break
            
            # Compute conflict score for each remaining candidate
            # Conflict score = number of k >= 2 such that k*x is achievable in dp_all
            best_candidate = None
            best_score = float('inf')
            
            for x in remaining:
                score = 0
                for k in range(2, n + 1):
                    target = k * x
                    if target > MAX_SUM:
                        break
                    if (dp_all[k] >> target) & 1:
                        score += 1
                        if score >= best_score:
                            break
                if score < best_score:
                    best_score = score
                    best_candidate = x
                    if score == 0:
                        break  # Can't do better than 0
            
            if best_candidate is None:
                break
            
            x = best_candidate
            
            # Check if x is valid (not mean of any subset of others)
            valid = True
            
            # Check 1: x should not be mean of any subset of selected elements
            for k in range(2, n + 1):
                target = k * x
                if target > MAX_SUM:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                remaining.discard(x)
                continue
            
            # Check 2: For each a in selected, a should not be mean of subset of (selected \ {a}) that includes x
            # Equivalent: (k+1)*a - x should not be achievable from selected \ {a} with k elements
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
                remaining.discard(x)
                continue
            
            # Valid, add x
            selected.append(x)
            remaining.discard(x)
            
            # Save old state
            old_dp_all = dp_all
            
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
            
            # For x, dp_without[x] = old_dp_all (padded)
            new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
            
            dp_without = new_dp_without
            n += 1
            
        return selected

    # Try multiple seeds for the conflict density greedy
    for seed in range(30):
        if time.time() - start_time > deadline:
            break
        res = try_greedy_conflict_density(seed)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
    
    # Also try some fixed orderings as fallback
    if best_len < 48:
        # Spread ordering
        spread = []
        lo, hi = 1, MAX_N
        while lo <= hi:
            spread.append(lo)
            if lo != hi:
                spread.append(hi)
            lo += 1
            hi -= 1
        
        def try_greedy_fixed(order):
            selected = []
            n = 0
            dp_all = [1]
            dp_without = {}
            
            for x in order:
                if time.time() - start_time > deadline:
                    break
                if n >= 48:
                    break
                
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
                
                selected.append(x)
                old_dp_all = dp_all
                
                new_dp_all = [0] * (n + 2)
                new_dp_all[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_all[k] if k <= n else 0
                    if k - 1 >= 0 and k - 1 <= n:
                        val |= (old_dp_all[k-1] << x)
                    new_dp_all[k] = val
                dp_all = new_dp_all
                
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
                
                new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
                dp_without = new_dp_without
                n += 1
            
            return selected
        
        res = try_greedy_fixed(spread)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
    
    return best_set

# EVOLVE-BLOCK-END
