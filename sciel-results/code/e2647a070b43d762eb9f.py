"""Combined approach: structural multiplier constraint + O(n) incremental local search + simulated annealing.

Structural idea (from current program): if nums = M * a_i, then every sum is M*(a_i+a_j),
so primes(M) are 'free' and we only pay for primes of the quotients. Pick M with few prime
factors and small a_i so the quotients are smooth.

Refinement machinery (from inspiration): maintain a prime->pair-count table so that evaluating
a single-element move is O(n) (only 11 sums change) instead of O(n^2). This ~10x speedup lets
us run simulated annealing and many more restarts within the 90s wall-clock budget.

New search family (structural change): simulated annealing with 1-move and 2-swap moves to
escape the single-move local minima that trap plain hill-climbing.
"""
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 82.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum = 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _sieve[_j] == _j:
                _sieve[_j] = _i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    if n <= 1:
        return frozenset()
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return frozenset(factors)

# Precompute factor sets for all sums 2..20000 (a sum of two distinct positives is >=3,
# but we fill 2..20000 to be safe). This makes incremental evaluation extremely fast.
_PFS = [frozenset() for _ in range(_MAX_SUM + 1)]
for _s in range(2, _MAX_SUM + 1):
    _PFS[_s] = _prime_factors_fast(_s)

def _build_count_table(nums):
    """Return dict prime -> number of pairs (i<j) whose sum is divisible by prime."""
    n = len(nums)
    cnt = {}
    for i in range(n):
        for j in range(i + 1, n):
            for p in _PFS[nums[i] + nums[j]]:
                cnt[p] = cnt.get(p, 0) + 1
    return cnt

def _score_from_count(cnt):
    return len(cnt)

def _score_full(nums):
    """Compute number of distinct prime factors across all pairwise sums."""
    n = len(nums)
    all_primes = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= _PFS[nums[i] + nums[j]]
    return len(all_primes)

def _valid(nums):
    return len(set(nums)) == 12 and all(1 <= x <= 10000 for x in nums)

def _try_one_move(nums, cnt, current_score, idx, new_val, rng):
    """Evaluate changing nums[idx] to new_val. Returns (new_score, new_cnt) or None if invalid."""
    if new_val in nums:
        return None
    n = len(nums)
    # New count table
    new_cnt = dict(cnt)
    # For each other element, remove old pair primes, add new pair primes
    for j in range(n):
        if j == idx:
            continue
        old_sum = nums[idx] + nums[j]
        new_sum = new_val + nums[j]
        for p in _PFS[old_sum]:
            new_cnt[p] -= 1
            if new_cnt[p] == 0:
                del new_cnt[p]
        for p in _PFS[new_sum]:
            new_cnt[p] = new_cnt.get(p, 0) + 1
    return len(new_cnt), new_cnt

def _local_search(nums, cnt, current_score, max_rounds=400):
    """Hill-climb with O(n) incremental moves. Returns (nums, cnt, score)."""
    n = len(nums)
    improved = True
    rounds = 0
    while improved and not _deadline_hit() and rounds < max_rounds:
        improved = False
        rounds += 1
        for idx in range(n):
            if _deadline_hit():
                break
            old_val = nums[idx]
            # Candidate values: small perturbations + a few randoms
            candidates = set()
            for d in range(-40, 41):
                v = old_val + d
                if 1 <= v <= 10000 and v != old_val:
                    candidates.add(v)
            for _ in range(8):
                candidates.add(random.randint(1, 10000))
            best_new_val = None
            best_new_score = current_score
            best_new_cnt = None
            for new_val in candidates:
                res = _try_one_move(nums, cnt, current_score, idx, new_val, random)
                if res is None:
                    continue
                new_score, new_cnt = res
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
                    best_new_cnt = new_cnt
            if best_new_val is not None:
                nums[idx] = best_new_val
                cnt = best_new_cnt
                current_score = best_new_score
                improved = True
                break
    return nums, cnt, current_score

def _simulated_annealing(nums, cnt, current_score, rng, steps=2000, T0=1.5, T1=0.05):
    """Simulated annealing with 1-move and 2-swap moves. Returns (nums, cnt, score)."""
    n = len(nums)
    best_nums = nums[:]
    best_cnt = dict(cnt)
    best_score = current_score
    cur_score = current_score
    cur_cnt = dict(cnt)
    cur_nums = nums[:]
    for step in range(steps):
        if _deadline_hit():
            break
        T = T0 * (T1 / T0) ** (step / max(1, steps))
        # Choose a move type
        if rng.random() < 0.7:
            # 1-move
            idx = rng.randrange(n)
            old_val = cur_nums[idx]
            # candidate: random or perturbation
            if rng.random() < 0.5:
                new_val = old_val + rng.randint(-30, 30)
            else:
                new_val = rng.randint(1, 10000)
            if new_val < 1 or new_val > 10000:
                new_val = rng.randint(1, 10000)
            res = _try_one_move(cur_nums, cur_cnt, cur_score, idx, new_val, rng)
            if res is None:
                continue
            new_score, new_cnt = res
            delta = new_score - cur_score
            if delta <= 0 or rng.random() < _exp_neg(delta / max(T, 1e-6)):
                # accept
                cur_nums[idx] = new_val
                cur_cnt = new_cnt
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_cnt = dict(cur_cnt)
                    best_nums = cur_nums[:]
        else:
            # 2-swap: change two elements
            i, j = rng.sample(range(n), 2)
            vi = cur_nums[i]
            vj = cur_nums[j]
            ni = vi + rng.randint(-30, 30)
            nj = vj + rng.randint(-30, 30)
            if ni < 1 or ni > 10000 or nj < 1 or nj > 10000:
                ni = rng.randint(1, 10000)
                nj = rng.randint(1, 10000)
            if ni in cur_nums or nj in cur_nums or ni == nj:
                continue
            # Evaluate 2-move: recompute count table (cheap enough, 66 pairs)
            new_nums = cur_nums[:]
            new_nums[i] = ni
            new_nums[j] = nj
            new_cnt = _build_count_table(new_nums)
            new_score = len(new_cnt)
            delta = new_score - cur_score
            if delta <= 0 or rng.random() < _exp_neg(delta / max(T, 1e-6)):
                cur_nums = new_nums
                cur_cnt = new_cnt
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_cnt = dict(cur_cnt)
                    best_nums = cur_nums[:]
    return best_nums, best_cnt, best_score

def _exp_neg(x):
    # safe exp(-x) for x>=0
    if x > 50:
        return 0.0
    import math
    return math.exp(-x)

def construct():
    random.seed(42)
    rng = random.Random(12345)

    n = 12
    best_nums = None
    best_score = float('inf')

    # Structural: multipliers with few prime factors
    multipliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]

    # Generate a pool of structural starting points
    starts = []
    for M in multipliers:
        max_a = 10000 // M
        if max_a < 12:
            continue
        # base 1..12
        base = [M * a for a in range(1, 13)]
        if _valid(base):
            starts.append(base[:])
        # random small a
        for _ in range(2):
            a_set = rng.sample(range(1, min(max_a + 1, 100)), 12)
            cand = [M * a for a in sorted(a_set)]
            if _valid(cand):
                starts.append(cand[:])
        # centered
        for center in [10, 20, 50]:
            if center + 6 > max_a:
                continue
            a_set = [center + i for i in range(-6, 7)]
            if all(a >= 1 for a in a_set):
                cand = [M * a for a in a_set]
                if _valid(cand):
                    starts.append(cand[:])
    # a few pure random starts
    for _ in range(10):
        cand = sorted(rng.sample(range(1, 10001), 12))
        starts.append(cand[:])

    # Evaluate all starts, keep the best few
    scored = []
    for cand in starts:
        if not _valid(cand):
            continue
        s = _score_full(cand)
        scored.append((s, cand))
    scored.sort(key=lambda x: x[0])

    # Run local search + SA from the top starting points
    for s0, cand in scored[:25]:
        if _deadline_hit():
            break
        nums = cand[:]
        cnt = _build_count_table(nums)
        cur_score = s0
        # local search first
        nums, cnt, cur_score = _local_search(nums, cnt, cur_score, max_rounds=200)
        # then SA
        nums, cnt, cur_score = _simulated_annealing(nums, cnt, cur_score, rng, steps=1500)
        if cur_score < best_score:
            best_score = cur_score
            best_nums = nums[:]
        if best_score <= 4:
            break

    # If nothing found (shouldn't happen), fallback
    if best_nums is None:
        best_nums = list(range(2, 25, 2))
        best_score = _score_full(best_nums)

    # Final validation
    assert len(set(best_nums)) == 12
    assert all(1 <= x <= 10000 for x in best_nums)

    return best_nums

# EVOLVE-BLOCK-END
