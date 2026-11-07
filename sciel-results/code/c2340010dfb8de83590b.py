import time
import random

def _init_dp(arr):
    n = len(arr)
    if n == 0:
        return [1]
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _is_valid(arr):
    n = len(arr)
    if n == 0:
        return True
    dp = _init_dp(arr)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _verify_add(dp, n, x):
    new_dp = [0] * (n + 2)
    for k in range(n + 1):
        new_dp[k] = dp[k]
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None
        acc |= new_dp[k]
    return new_dp

def _grow_greedy(start, strategy='smallest', deadline=None, rng=None):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _init_dp(cur)
    
    improved = True
    while improved:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved = False
        candidates = [v for v in range(1, 257) if v not in cur_set]
        if not candidates:
            break
        if strategy == 'smallest':
            order = candidates
        elif strategy == 'largest':
            order = list(reversed(candidates))
        elif strategy == 'random':
            order = candidates[:]
            if rng:
                rng.shuffle(order)
            else:
                random.shuffle(order)
        else:
            order = candidates
        for v in order:
            new_dp = _verify_add(dp, n, v)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                n += 1
                dp = new_dp
                improved = True
                break
    return cur

def _search_consecutive():
    # Find largest n such that AP is valid
    # We know AP is valid if n is small enough or shift is large enough
    # But we want to use this as a START for greedy
    best = []
    for n in range(60, 0, -1):
        # Try various shifts
        for a in range(1, 257 - n + 1):
            arr = list(range(a, a + n))
            if _is_valid(arr):
                if len(arr) > len(best):
                    best = arr
                # If we find one, we can break if we want just one, 
                # but maybe a larger n with different a is better?
                # Actually, if n is valid, smaller n are also valid (subset).
                # So we just need the max n.
                break
        if len(best) == n:
            break
    return best

def _polish(cur, deadline):
    cur = list(cur)
    n = len(cur)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        cur_set = set(cur)
        for i in range(n):
            old = cur[i]
            remaining = cur[:i] + cur[i+1:]
            remaining_set = set(remaining)
            rem_dp = _init_dp(remaining)
            # Try replacing with smaller unused values
            for v in range(1, old):
                if v in remaining_set:
                    continue
                new_dp = _verify_add(rem_dp, n - 1, v)
                if new_dp is not None:
                    cur_new = remaining + [v]
                    cur_new.sort()
                    grown = _grow_greedy(cur_new, 'smallest', deadline)
                    if len(grown) > len(cur):
                        cur = grown
                        n = len(cur)
                        improved = True
                        break
            if improved:
                break
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0
    best = None
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
    
    # 1. Analytical consecutive search
    res = _search_consecutive()
    if res:
        update_best(res)
    
    # 2. Greedy searches
    if best is None or len(best) < 40:
        res = _grow_greedy([], 'smallest', deadline)
        update_best(res)
        res = _grow_greedy([], 'largest', deadline)
        update_best(res)
        
        p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        res = _grow_greedy(p2, 'smallest', deadline)
        update_best(res)
        res = _grow_greedy(p2, 'largest', deadline)
        update_best(res)
        
        # Randomized restarts
        rng = random.Random(42)
        if time.monotonic() < deadline - 5:
            for _ in range(200):
                if time.monotonic() > deadline - 2:
                    break
                k = rng.randint(0, 5)
                if k == 0:
                    start = []
                else:
                    start = rng.sample(range(1, 257), k)
                    if not _is_valid(start):
                        continue
                strategy = rng.choice(['smallest', 'largest', 'random'])
                res = _grow_greedy(start, strategy, deadline, rng)
                update_best(res)
    
    # 3. Polish
    if best is not None and time.monotonic() < deadline - 3:
        polished = _polish(best, deadline)
        if len(polished) > len(best):
            best = polished
    
    # Final validity check
    if best is not None:
        if not _is_valid(best):
            best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best
