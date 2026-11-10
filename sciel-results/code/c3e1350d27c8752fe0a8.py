"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000.

Strategy: For a small set S of primes, precompute all S-smooth numbers <= 10000,
build a compatibility graph (edge if sum is S-smooth), then find a clique of size 12
via backtracking. Try many small prime sets. Also do local search as fallback.
"""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0  # safety margin under 90s wall-clock

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

def _prime_factors_fast(n):
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

_pf_cache = [None] * (_MAX_SUM + 1)
def _pf(s):
    r = _pf_cache[s]
    if r is None:
        r = _prime_factors_fast(s)
        _pf_cache[s] = r
    return r

N = 12
MAX_V = 10000

def _is_smooth(n, S):
    """Check if all prime factors of n are in S."""
    x = n
    while x > 1:
        p = _sieve[x]
        if p not in S:
            return False
        while x % p == 0:
            x //= p
    return True

def _smooth_numbers(S):
    """Return sorted list of all S-smooth numbers in [1, MAX_V]."""
    result = []
    for n in range(1, MAX_V + 1):
        if _is_smooth(n, S):
            result.append(n)
    return result

def _find_clique(nodes, adj, size, deadline_fn):
    """Backtracking to find a clique of given size. Returns list or None."""
    if len(nodes) < size:
        return None
    if size == 0:
        return []
    
    # Order nodes by degree in the induced subgraph (descending)
    # Use Bron-Kerbosch style with pivot
    def _bk(R, P, X, remaining):
        if deadline_fn():
            return None
        if remaining == 0:
            return R[:]
        if not P:
            return None
        
        # Pivot: choose u in P ∪ X with max degree in P
        pivot_pool = list(P) + list(X)
        if not pivot_pool:
            return None
        # Pick pivot with max neighbors in P
        best_pivot = max(pivot_pool, key=lambda u: len(adj[u] & P))
        candidates = [v for v in P if v not in adj[best_pivot]]
        
        for v in candidates:
            if deadline_fn():
                return None
            new_R = R + [v]
            new_P = P & adj[v]
            new_X = X & adj[v]
            result = _bk(new_R, new_P, new_X, remaining - 1)
            if result is not None:
                return result
            P.discard(v)
            X.add(v)
        return None
    
    return _bk([], set(nodes), set(), size)

def _try_prime_set(S, rng, max_nodes=None):
    """Try to find 12 distinct integers whose pairwise sums are all S-smooth.
    Returns (score, nums) where score is number of distinct primes in sums."""
    smooth = _smooth_numbers(S)
    if len(smooth) < N:
        return None, None
    
    if max_nodes and len(smooth) > max_nodes:
        # Sample or use all - use all for correctness but limit backtracking
        pass
    
    # Build adjacency: i-j edge if smooth[i] + smooth[j] is S-smooth
    m = len(smooth)
    # For speed, if m is large, subsample
    if m > 500:
        # Use a random subset of 500 for initial attempt
        indices = list(range(m))
        rng.shuffle(indices)
        indices = indices[:500]
        smooth = [smooth[i] for i in indices]
        m = 500
    
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if _is_smooth(smooth[i] + smooth[j], S):
                adj[i].add(j)
                adj[j].add(i)
    
    # Find clique of size N
    clique = _find_clique(range(m), adj, N, _deadline_hit)
    if clique is not None:
        nums = [smooth[i] for i in clique]
        # Verify
        primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                primes |= _pf(nums[i] + nums[j])
        return len(primes), nums
    
    return None, None

def _objective(nums):
    primes = set()
    for i in range(N):
        for j in range(i + 1, N):
            primes |= _pf(nums[i] + nums[j])
    return len(primes)

def _local_search(nums, rng, iters, T0=30.0, T1=0.01):
    """Simulated annealing to minimize distinct prime count."""
    cur = nums[:]
    best = nums[:]
    best_obj = _objective(best)
    
    def _total_appearances(arr):
        cnt = 0
        for i in range(N):
            for j in range(i + 1, N):
                cnt += len(_pf(arr[i] + arr[j]))
        return cnt
    
    cur_cnt = _total_appearances(cur)
    best_cnt = cur_cnt
    
    for it in range(iters):
        if _deadline_hit():
            break
        T = T0 * (T1 / T0) ** (it / max(1, iters))
        idx = rng.randint(0, N - 1)
        if rng.random() < 0.7:
            delta = rng.randint(-200, 200)
            new_val = cur[idx] + delta
        else:
            new_val = rng.randint(1, MAX_V)
        new_val = max(1, min(MAX_V, new_val))
        if new_val == cur[idx]:
            continue
        if any(cur[j] == new_val for j in range(N) if j != idx):
            continue
        
        old_val = cur[idx]
        old_cnt = sum(len(_pf(old_val + cur[j])) for j in range(N) if j != idx)
        new_cnt = sum(len(_pf(new_val + cur[j])) for j in range(N) if j != idx)
        delta_cnt = new_cnt - old_cnt
        
        if delta_cnt < 0 or rng.random() < math.exp(-delta_cnt / max(T, 1e-9)):
            cur[idx] = new_val
            cur_cnt += delta_cnt
            if cur_cnt < best_cnt:
                best_cnt = cur_cnt
                obj = _objective(cur)
                if obj < best_obj:
                    best_obj = obj
                    best = cur[:]
    
    return best_obj, best

def construct():
    rng = random.Random(42)
    best_score = 10 ** 9
    best_nums = list(range(1, 13))
    
    # Generate list of small primes
    small_primes = []
    m = 2
    while len(small_primes) < 50:
        if _sieve[m] == m:
            small_primes.append(m)
        m += 1
    
    # Phase 1: Try small prime sets via clique finding
    # Try sets of size 3, 4, 5, 6
    prime_sets_to_try = []
    
    # All combinations of small primes of size 3-6
    for k in range(3, 7):
        for combo in combinations(small_primes[:20], k):
            prime_sets_to_try.append(set(combo))
    
    # Also try the first k smallest primes
    for k in range(3, 15):
        prime_sets_to_try.append(set(small_primes[:k]))
    
    # Shuffle for diversity
    rng.shuffle(prime_sets_to_try)
    
    found_good = False
    for S in prime_sets_to_try:
        if _deadline_hit():
            break
        score, nums = _try_prime_set(S, rng)
        if nums is not None and score is not None:
            if score < best_score:
                best_score = score
                best_nums = nums[:]
            if score <= 4:
                found_good = True
                # Continue trying to find even better
                if score == 3:
                    break  # Can't do better than 3 for 12 distinct positive ints
    
    # Phase 2: Local search from multiple seeds
    inits = [
        list(range(1, 13)),
        list(range(2, 25, 2)),
        list(range(6, 78, 6)),
        list(range(12, 156, 12)),
        [2 ** i for i in range(1, 13)],
        [3 ** i for i in range(1, 13)],
    ]
    for _ in range(10):
        inits.append(sorted(rng.sample(range(1, MAX_V + 1), N)))
    
    # Add best_nums as a seed
    inits.append(best_nums[:])
    
    for start in inits:
        if _deadline_hit():
            break
        if len(set(start)) != N:
            continue
        if not all(1 <= x <= MAX_V for x in start):
            continue
        obj, nums = _local_search(start, rng, 50000, 50.0, 0.01)
        if obj < best_score:
            best_score = obj
            best_nums = nums[:]
    
    # Phase 3: Refine - try to reduce prime set by checking which primes are actually needed
    # and do targeted local search
    if best_score > 3 and not _deadline_hit():
        # Extract current prime set
        cur_primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                cur_primes |= _pf(best_nums[i] + best_nums[j])
        
        # Try local search with bias toward S-smooth values
        smooth = _smooth_numbers(cur_primes)
        if len(smooth) >= N:
            # Start from smooth numbers
            rng.shuffle(smooth)
            start = smooth[:N]
            # Ensure distinct
            if len(set(start)) == N:
                obj, nums = _local_search(start, rng, 30000, 20.0, 0.005)
                if obj < best_score:
                    best_score = obj
                    best_nums = nums[:]
    
    # Phase 4: Final polish - try small perturbations
    improved = True
    while improved and not _deadline_hit():
        improved = False
        cur = best_nums[:]
        cur_score = best_score
        for idx in range(N):
            if _deadline_hit():
                break
            old_val = cur[idx]
            for d in range(-50, 51, 2):
                v = old_val + d
                if 1 <= v <= MAX_V and v != old_val:
                    trial = cur[:]
                    trial[idx] = v
                    if len(set(trial)) != N:
                        continue
                    obj = _objective(trial)
                    if obj < cur_score:
                        cur = trial
                        cur_score = obj
                        improved = True
                        break
            if improved:
                break
        best_nums = cur
        best_score = cur_score
    
    result = best_nums
    assert len(set(result)) == N
    assert all(1 <= x <= MAX_V for x in result)
    return result
