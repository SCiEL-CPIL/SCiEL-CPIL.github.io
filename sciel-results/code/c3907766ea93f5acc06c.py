"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import random
from collections import defaultdict

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum is 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _sieve[_j] == _j:
                _sieve[_j] = _i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _compute_pair_factors(nums):
    """Compute factor sets for all pairs. Returns 12x12 matrix of sets."""
    n = len(nums)
    pf = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf_set = _prime_factors_fast(s)
            pf[i][j] = pf_set
            pf[j][i] = pf_set
    return pf

def _total_primes(pf):
    """Count distinct primes across all pair factor sets."""
    all_primes = set()
    n = len(pf)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pf[i][j]
    return len(all_primes)

def _build_prime_count(pf):
    """Build dict mapping each prime to number of pairs containing it."""
    n = len(pf)
    pc = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            for p in pf[i][j]:
                pc[p] += 1
    return pc

def construct():
    random.seed(42)
    n = 12

    # Generate diverse initial candidates with structural properties
    candidates = []

    # Multiples of small numbers (so sums share common factors)
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60]:
        if m * n <= 10000:
            # Spread out: m, 2m, ..., nm
            candidates.append([m * (i + 1) for i in range(n)])
            # Or clustered: m, m, 2m, 2m, ... but need distinct
            # Use m*(1), m*(2), ... m*(n) with some offset
            candidates.append([m * (i + 1) + m // 2 for i in range(n)] if m * (n + 1) + m // 2 <= 10000 else None)

    # Powers of 2
    candidates.append([2 ** i for i in range(1, 13)])
    candidates.append([2 ** i for i in range(0, 12)])

    # Small consecutive
    candidates.append(list(range(1, 13)))
    candidates.append(list(range(2, 14)))
    candidates.append(list(range(10, 22)))

    # Multiples with gaps
    for base in [1, 2, 3, 5, 7, 10]:
        for step in [1, 2, 3, 4, 5, 6, 8, 10]:
            cand = [base + i * step for i in range(n)]
            if max(cand) <= 10000:
                candidates.append(cand)

    # Random candidates
    for _ in range(20):
        candidates.append(sorted(random.sample(range(1, 10001), n)))

    # Filter valid candidates
    valid_candidates = []
    for c in candidates:
        if c is None:
            continue
        if len(c) != n:
            continue
        if len(set(c)) != n:
            continue
        if not all(1 <= x <= 10000 for x in c):
            continue
        valid_candidates.append(c)

    if not valid_candidates:
        valid_candidates = [list(range(1, 13))]

    # Evaluate all candidates and find best
    best_candidate = None
    best_score = float('inf')
    best_pf = None

    for cand in valid_candidates:
        if _deadline_hit():
            break
        pf = _compute_pair_factors(cand)
        score = _total_primes(pf)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
            best_pf = pf

    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_pf = _compute_pair_factors(best_candidate)
        best_score = _total_primes(best_pf)

    current = best_candidate[:]
    current_pf = best_pf
    current_score = best_score
    prime_count = _build_prime_count(current_pf)

    def _try_replace(idx, new_val):
        """Try replacing current[idx] with new_val. Returns True if accepted."""
        nonlocal current, current_pf, current_score, prime_count
        if new_val in current:
            return False
        if not (1 <= new_val <= 10000):
            return False

        # Compute old and new prime sets for pairs involving idx
        old_primes_set = set()
        new_primes_set = set()
        for j in range(n):
            if j == idx:
                continue
            old_primes_set |= current_pf[idx][j]
            new_primes_set |= _prime_factors_fast(new_val + current[j])

        # Compute delta in distinct prime count
        delta = 0
        for p in old_primes_set:
            if prime_count.get(p, 0) == 1:
                delta -= 1
        for p in new_primes_set:
            if prime_count.get(p, 0) == 0:
                delta += 1

        new_score = current_score + delta
        if new_score < current_score:
            # Accept
            for j in range(n):
                if j == idx:
                    continue
                old_pf_set = current_pf[idx][j]
                new_pf_set = _prime_factors_fast(new_val + current[j])
                for p in old_pf_set:
                    prime_count[p] -= 1
                    if prime_count[p] == 0:
                        del prime_count[p]
                for p in new_pf_set:
                    if p not in prime_count:
                        prime_count[p] = 0
                    prime_count[p] += 1
                current_pf[idx][j] = new_pf_set
                current_pf[j][idx] = new_pf_set
            current[idx] = new_val
            current_score = new_score
            return True
        return False

    def _try_swap(i, j):
        """Try swapping current[i] and current[j]. Returns True if accepted."""
        nonlocal current, current_pf, current_score, prime_count
        if i == j:
            return False
        vi, vj = current[i], current[j]
        if vi == vj:
            return False

        # Pairs affected: (i,k) and (j,k) for all k != i,j, plus (i,j)
        old_primes_all = set()
        new_primes_all = set()
        for k in range(n):
            if k == i or k == j:
                continue
            # Old: (i,k) has vi+current[k], (j,k) has vj+current[k]
            old_primes_all |= current_pf[i][k]
            old_primes_all |= current_pf[j][k]
            # New: (i,k) has vj+current[k], (j,k) has vi+current[k]
            new_primes_all |= _prime_factors_fast(vj + current[k])
            new_primes_all |= _prime_factors_fast(vi + current[k])
        # Pair (i,j): old vi+vj, new vj+vi (same!)
        old_primes_all |= current_pf[i][j]
        new_primes_all |= current_pf[i][j]

        delta = 0
        for p in old_primes_all:
            if prime_count.get(p, 0) == 1:
                delta -= 1
        for p in new_primes_all:
            if prime_count.get(p, 0) == 0:
                delta += 1

        new_score = current_score + delta
        if new_score < current_score:
            # Accept swap
            for k in range(n):
                if k == i or k == j:
                    continue
                # Update (i,k)
                old_pf_ik = current_pf[i][k]
                new_pf_ik = _prime_factors_fast(vj + current[k])
                for p in old_pf_ik:
                    prime_count[p] -= 1
                    if prime_count[p] == 0:
                        del prime_count[p]
                for p in new_pf_ik:
                    if p not in prime_count:
                        prime_count[p] = 0
                    prime_count[p] += 1
                current_pf[i][k] = new_pf_ik
                current_pf[k][i] = new_pf_ik

                # Update (j,k)
                old_pf_jk = current_pf[j][k]
                new_pf_jk = _prime_factors_fast(vi + current[k])
                for p in old_pf_jk:
                    prime_count[p] -= 1
                    if prime_count[p] == 0:
                        del prime_count[p]
                for p in new_pf_jk:
                    if p not in prime_count:
                        prime_count[p] = 0
                    prime_count[p] += 1
                current_pf[j][k] = new_pf_jk
                current_pf[k][j] = new_pf_jk
            # Pair (i,j) unchanged in value but we swap indices
            current[i], current[j] = vj, vi
            current_score = new_score
            return True
        return False

    # Main local search loop with multiple restarts
    for restart in range(500):
        if _deadline_hit():
            break

        improved = True
        while improved and not _deadline_hit():
            improved = False

            # Try single-element replacements with structured perturbations
            for idx in range(n):
                if _deadline_hit():
                    break
                old_val = current[idx]
                tried = set([old_val])

                # Small perturbations
                for d in range(-20, 21):
                    v = old_val + d
                    if 1 <= v <= 10000 and v not in current and v not in tried:
                        tried.add(v)
                        if _try_replace(idx, v):
                            improved = True
                            break
                if improved:
                    break

                # Medium perturbations
                for d in range(-50, 51, 2):
                    v = old_val + d
                    if 1 <= v <= 10000 and v not in current and v not in tried:
                        tried.add(v)
                        if _try_replace(idx, v):
                            improved = True
                            break
                if improved:
                    break

                # Random values
                for _ in range(10):
                    v = random.randint(1, 10000)
                    if v not in current and v not in tried:
                        tried.add(v)
                        if _try_replace(idx, v):
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # Try swaps
            for i in range(n):
                if _deadline_hit():
                    break
                for j in range(i + 1, n):
                    if _deadline_hit():
                        break
                    if _try_swap(i, j):
                        improved = True
                        break
                if improved:
                    break

        # If stuck, try a random perturbation to escape local minimum
        if not _deadline_hit():
            new_cand = current[:]
            for _ in range(random.randint(1, 4)):
                idx = random.randint(0, n - 1)
                new_cand[idx] = random.randint(1, 10000)
            if len(set(new_cand)) == n and all(1 <= x <= 10000 for x in new_cand):
                new_pf = _compute_pair_factors(new_cand)
                new_score = _total_primes(new_pf)
                if new_score < current_score:
                    current = new_cand
                    current_pf = new_pf
                    current_score = new_score
                    prime_count = _build_prime_count(current_pf)
                elif random.random() < 0.1:  # Occasionally accept worse to escape
                    current = new_cand
                    current_pf = new_pf
                    current_score = new_score
                    prime_count = _build_prime_count(current_pf)

    # Final polish: tighter local search
    if not _deadline_hit():
        improved = True
        while improved and not _deadline_hit():
            improved = False
            for idx in range(n):
                if _deadline_hit():
                    break
                old_val = current[idx]
                for d in range(-5, 6):
                    v = old_val + d
                    if 1 <= v <= 10000 and v != old_val and v not in current:
                        if _try_replace(idx, v):
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            for i in range(n):
                if _deadline_hit():
                    break
                for j in range(i + 1, n):
                    if _deadline_hit():
                        break
                    if _try_swap(i, j):
                        improved = True
                        break
                if improved:
                    break

    # Final verification
    result = current
    assert len(set(result)) == n
    assert all(1 <= x <= 10000 for x in result)

    return result
