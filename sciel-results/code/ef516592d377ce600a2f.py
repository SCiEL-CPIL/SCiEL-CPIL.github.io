import time
import math
import random
import numpy as np

def construct():
    _t0 = time.time()
    _deadline = _t0 + 80.0  # safety margin before 90s kill

    # primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    k = len(primes)
    N = 24

    def extract(x):
        f = 1
        for p in primes:
            while x % p == 0:
                f *= p
                x //= p
        return f

    def score_exact(n):
        vals = [extract(n + i) for i in range(1, N + 1)]
        return len(set(vals))

    # ---------- initial best ----------
    best_n = 1000
    best_score = score_exact(best_n)

    # quick scan near 1000 to get a decent baseline
    for n in range(1000, 1030):
        s = score_exact(n)
        if s < best_score:
            best_score = s
            best_n = n

    # ---------- vectorized candidate evaluation ----------
    def evaluate_batch(cands):
        """Return list of (score, n) for candidate n values, using vectorized extraction."""
        results = []
        for n in cands:
            # compute the 24 extracted factors
            arr = np.array([n + i for i in range(1, N + 1)], dtype=np.int64)
            extracted = np.ones(N, dtype=np.int64)
            for p in primes:
                tmp = arr.copy()
                while np.any(tmp % p == 0):
                    mask = (tmp % p == 0)
                    if not mask.any():
                        break
                    extracted[mask] *= p
                    tmp[mask] //= p
            # count distinct values
            uniq = np.unique(extracted)
            s = len(uniq)
            results.append((s, n))
        return results

    # ---------- local search around a candidate ----------
    def local_search(start_n, budget):
        nonlocal best_n, best_score
        cur_n = start_n
        cur_s = score_exact(cur_n)
        if cur_s < best_score:
            best_score = cur_s
            best_n = cur_n
        t = time.time()
        step_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
        while time.time() < budget:
            # pick a random step
            if random.random() < 0.7:
                d = random.choice(step_sizes[:8])
            else:
                d = random.choice(step_sizes)
            direction = random.choice([-1, 1])
            new_n = cur_n + d * direction
            if new_n < 1000 or new_n > 1000000:
                continue
            new_s = score_exact(new_n)
            if new_s <= cur_s:
                cur_n = new_n
                cur_s = new_s
                if cur_s < best_score:
                    best_score = cur_s
                    best_n = cur_n
            # small probability of big jump
            if random.random() < 0.05:
                jump = random.randint(-500, 500)
                new_n = cur_n + jump
                if 1000 <= new_n <= 1000000:
                    new_s = score_exact(new_n)
                    if new_s <= cur_s:
                        cur_n = new_n
                        cur_s = new_s
                        if cur_s < best_score:
                            best_score = cur_s
                            best_n = cur_n
        return best_n, best_score

    # ---------- random search with batch evaluation ----------
    def random_search(budget):
        nonlocal best_n, best_score
        t = time.time()
        batch_size = 200
        while time.time() < budget:
            # generate candidates in a range
            lo = random.randint(1000, 999000)
            hi = min(lo + 10000, 1000000)
            cands = list(set(random.sample(range(lo, hi), min(batch_size, hi - lo))))
            results = evaluate_batch(cands)
            for s, n in results:
                if s < best_score:
                    best_score = s
                    best_n = n
            # if we found something good, do a quick local search
            if best_score <= 3:
                local_search(best_n, time.time() + 5.0)
        return best_n, best_score

    # ---------- structured search: look for numbers with many small-prime factors ----------
    def structured_search(budget):
        nonlocal best_n, best_score
        t = time.time()
        # Try to find n where n+1..n+24 are all highly composite w.r.t. small primes
        # Strategy: for each i in 1..24, we want n+i to have few distinct small-prime factors
        # A number with only prime 2 as small-prime factor is a power of 2 times a large-prime part
        # So look for n such that n+i = 2^a_i * m_i where m_i has no small prime factors
        # This means n+i is "2-smooth" part times "large" part
        
        # Simpler: scan regions where numbers tend to be smooth
        # Check multiples of small lcms
        for base in range(1000, 1000000, 5000):
            if time.time() > budget:
                break
            # look at a window
            window = list(range(base, min(base + 500, 1000000)))
            results = evaluate_batch(window)
            for s, n in results:
                if s < best_score:
                    best_score = s
                    best_n = n
        return best_n, best_score

    # ---------- smooth surrogate optimization (epigraph) ----------
    def smooth_optimize(budget):
        """
        Use a smooth surrogate: for each n, compute the 24 extracted factors,
        then use a soft-min to encourage them to be equal.
        We minimize the "spread" of the extracted factors.
        """
        nonlocal best_n, best_score
        t = time.time()
        
        def smooth_score(n):
            vals = np.array([extract(n + i) for i in range(1, N + 1)], dtype=np.float64)
            # soft-min: we want all vals to be equal
            # Use log-sum-exp trick: if all vals equal v, then logsumexp(vals) = log(N) + log(v)
            # The "excess" is logsumexp(vals) - log(N) - log(min(vals))
            # But we want to minimize distinct count, so maximize agreement
            # Alternative: use variance as a smooth proxy
            mean = np.mean(vals)
            var = np.var(vals)
            # Also consider the range
            rng = np.max(vals) - np.min(vals)
            return var + rng
        
        # Start from current best and do gradient-free optimization
        cur_n = best_n
        cur_smooth = smooth_score(cur_n)
        
        while time.time() < budget:
            # Try perturbations
            improved = False
            for _ in range(50):
                if time.time() > budget:
                    break
                d = random.randint(1, 20)
                direction = random.choice([-1, 1])
                new_n = cur_n + d * direction
                if new_n < 1000 or new_n > 1000000:
                    continue
                new_smooth = smooth_score(new_n)
                if new_smooth < cur_smooth:
                    cur_n = new_n
                    cur_smooth = new_smooth
                    # Check exact score
                    exact_s = score_exact(cur_n)
                    if exact_s < best_score:
                        best_score = exact_s
                        best_n = cur_n
                    improved = True
            if not improved:
                # Try a bigger jump
                d = random.randint(100, 1000)
                direction = random.choice([-1, 1])
                new_n = cur_n + d * direction
                if 1000 <= new_n <= 1000000:
                    new_smooth = smooth_score(new_n)
                    if new_smooth < cur_smooth:
                        cur_n = new_n
                        cur_smooth = new_smooth
                        exact_s = score_exact(cur_n)
                        if exact_s < best_score:
                            best_score = exact_s
                            best_n = cur_n
        
        return best_n, best_score

    # ---------- main search loop ----------
    # Phase 1: quick structured scan
    structured_search(_t0 + 20.0)
    
    # Phase 2: local search from best
    local_search(best_n, _t0 + 40.0)
    
    # Phase 3: smooth optimization
    smooth_optimize(_t0 + 60.0)
    
    # Phase 4: final local search
    local_search(best_n, _deadline)
    
    return best_n
