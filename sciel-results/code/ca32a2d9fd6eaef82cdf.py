# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations, permutations

def _set_sum_diff(U):
    """Return (|U+U|, |U-U|) for a finite set U of nonnegative ints."""
    Us = sorted(set(int(x) for x in U))
    n = len(Us)
    if n == 0:
        return 0, 0
    plus = set()
    minus = set()
    for i in range(n):
        a = Us[i]
        for j in range(i, n):
            plus.add(a + Us[j])
        for j in range(n):
            minus.add(a - Us[j])
    return len(plus), len(minus)

def _score(U):
    """Lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U or len(U) <= 1:
        return 0.0, sorted(U)
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
    nplus, nminus = _set_sum_diff(U)
    if nplus == 0 or nminus == 0:
        return 0.0, sorted(U)
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, sorted(U)
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    rng = random.Random(12345)
    
    best_val = -1e9
    best_set = [0]
    
    def consider(U):
        nonlocal best_val, best_set
        if time.time() > deadline:
            return
        try:
            U = list(set(int(x) for x in U))
            if 0 not in U:
                U.append(0)
            if len(U) <= 1:
                return
            v, s = _score(U)
            if v > best_val + 1e-12:
                best_val = v
                best_set = s
        except Exception:
            pass

    # 1. Exhaustive search for small sets
    # The theoretical max for small n is bounded. 
    # For n=4, {0,1,3,7} or similar lacunary sets might be good.
    # Let's search up to max element 30 for n <= 5.
    for max_val in range(1, 31):
        for n in range(2, 6):
            for combo in combinations(range(1, max_val + 1), n - 1):
                U = [0] + list(combo)
                consider(U)
        if time.time() > deadline - 60:
            break

    # 2. Structured families
    # Lacunary sets: {0, b, b^2, ...}
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for n in range(2, 20):
            U = [base**k for k in range(n)]
            if max(U) < 10**9: # Safety
                consider(U)
    
    # Modified lacunary: {0, b, b^2+1, b^3-1, ...}
    for base in [2, 3, 5]:
        for n in range(2, 10):
            base_set = [base**k for k in range(n)]
            for offsets in [[0, 1, -1], [0, -1, 1], [1, 1, 1], [-1, -1, -1]]:
                U = [0]
                valid = True
                for i in range(1, n):
                    off = offsets[i % 3]
                    val = base**i + off
                    if val <= 0 or val in U:
                        valid = False
                        break
                    U.append(val)
                if valid:
                    consider(U)

    # 3. Random search with local optimization
    cur_best = list(best_set)
    
    # Phase 3a: Random restarts
    for _ in range(5000):
        if time.time() > deadline - 10:
            break
        n = rng.randint(3, 12)
        # Use a range that scales
        maxv = rng.choice([50, 100, 200, 500, 1000, 2000, 5000])
        vals = rng.sample(range(1, maxv + 1), min(n - 1, maxv))
        U = [0] + vals
        consider(U)

    # Phase 3b: Local search on the best found
    cur = list(best_set)
    if not cur or len(cur) == 1:
        cur = [0, 1, 3]
    
    improved = True
    iters = 0
    max_iters = 10000
    while improved and time.time() < deadline - 2.0 and iters < max_iters:
        improved = False
        iters += 1
        mx = max(cur)
        
        # Candidate generation
        cands = set()
        # Small additions
        for d in range(1, 10):
            cands.add(mx + d)
        # Geometric additions
        cands.add(mx * 2)
        cands.add(mx * 2 + 1)
        cands.add(mx * 3)
        cands.add(mx * 3 + 1)
        cands.add(max(1, mx // 2))
        cands.add(max(1, mx // 3))
        # Random additions
        for _ in range(20):
            cands.add(rng.randint(1, mx * 4 + 10))
        
        # Filter valid candidates
        cands = [c for c in cands if c > 0 and c not in cur]
        rng.shuffle(cands)
        
        # Try adding
        for c in cands:
            U = cur + [c]
            v, s = _score(U)
            if v > best_val + 1e-12:
                best_val = v
                best_set = s
                cur = s
                improved = True
                break
        
        # Try removing
        if not improved and len(cur) > 2:
            for i in range(1, len(cur)):
                U = [x for x in cur if x != cur[i]]
                v, s = _score(U)
                if v > best_val + 1e-12:
                    best_val = v
                    best_set = s
                    cur = s
                    improved = True
                    break
                    
        # Try swapping (replace one element with another)
        if not improved and len(cur) > 2:
            for i in range(1, len(cur)):
                # Generate neighbors for cur[i]
                val = cur[i]
                for c in [val-1, val+1, val*2, val//2, val+10, val-10]:
                    if c <= 0 or c in cur:
                        continue
                    U = list(cur)
                    U[i] = c
                    v, s = _score(U)
                    if v > best_val + 1e-12:
                        best_val = v
                        best_set = s
                        cur = s
                        improved = True
                        break
                if improved:
                    break

    # Phase 3c: Hill climbing with random restarts on local optima
    if time.time() < deadline - 1.0:
        for _ in range(2000):
            if time.time() > deadline - 0.5:
                break
            # Start from best or a random perturbation of best
            if rng.random() < 0.5:
                cur = list(best_set)
            else:
                cur = list(best_set)
                # Randomly remove 1-2 elements
                for _ in range(rng.randint(1, 2)):
                    if len(cur) > 2:
                        idx = rng.randint(1, len(cur)-1)
                        cur.pop(idx)
                
            # Quick local search
            for _ in range(50):
                if time.time() > deadline - 0.5:
                    break
                mx = max(cur) if cur else 1
                cands = set()
                for d in range(1, 5):
                    cands.add(mx + d)
                cands.add(mx * 2)
                cands.add(mx * 2 + 1)
                for _ in range(10):
                    cands.add(rng.randint(1, mx * 3))
                cands = [c for c in cands if c > 0 and c not in cur]
                rng.shuffle(cands)
                
                found = False
                for c in cands:
                    U = cur + [c]
                    v, s = _score(U)
                    if v > best_val + 1e-12:
                        best_val = v
                        best_set = s
                        cur = s
                        found = True
                        break
                if not found:
                    # Try removal
                    if len(cur) > 2:
                        idx = rng.randint(1, len(cur)-1)
                        U = [x for x in cur if x != cur[idx]]
                        v, s = _score(U)
                        if v > best_val + 1e-12:
                            best_val = v
                            best_set = s
                            cur = s

    return best_set
# EVOLVE-BLOCK-END
