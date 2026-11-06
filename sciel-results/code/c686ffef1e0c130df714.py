# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_counts(residues, primes, max_check=100000):
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _score_from_counts(counts, max_check=100000):
    below = np.nonzero(counts[1:] < 2)[0]
    if len(below) == 0:
        return max_check
    return int(below[0])

def _score(residues, primes, max_check=100000):
    counts = _compute_counts(residues, primes, max_check)
    return _score_from_counts(counts, max_check)

def _fast_score(counts, best_score, max_check=100000):
    # If best_score already has count>=2, only need to scan [best_score, next_gap).
    # Return the new score.
    if counts[best_score] < 2:
        return best_score - 1
    # find next gap starting from best_score
    seg = counts[best_score:]
    below = np.nonzero(seg < 2)[0]
    if len(below) == 0:
        return max_check
    return best_score + int(below[0])

def _full_search(seed_residues, primes, deadline, max_check=100000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(primes)
    residues = list(seed_residues)
    counts = _compute_counts(residues, primes, max_check)
    best_score = _score_from_counts(counts, max_check)
    best_residues = list(residues)

    # Precompute per-prime candidate lists
    # For small primes: all residues. For large primes: a sample.
    all_candidates = []
    for i, p in enumerate(primes):
        if p <= 40:
            cands = list(range(p))
        else:
            # sample: 0, small values, and randoms
            c = set([0, 1, 2, 3, 4, 5, 6])
            c.update(int(x) for x in rng.integers(0, p, size=20))
            cands = list(c)
        all_candidates.append(cands)

    # Greedy local search: iterate over primes, try all candidates, keep best.
    # Repeat until no improvement or time up.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return best_residues, best_score
            p = primes[i]
            cur = residues[i]
            old_start = cur if cur > 0 else p
            best_local_score = best_score
            best_local_r = cur
            for new_r in all_candidates[i]:
                if new_r == cur:
                    continue
                new_start = new_r if new_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                if new_start <= max_check:
                    counts[new_start::p] += 1
                s = _fast_score(counts, best_score, max_check)
                if s > best_local_score:
                    best_local_score = s
                    best_local_r = new_r
                else:
                    if new_start <= max_check:
                        counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
            if best_local_r != cur:
                # apply
                residues[i] = best_local_r
                best_score = best_local_score
                improved = True
            # else: counts already reverted to original state
    return best_residues, best_score

def _multi_residue_moves(residues, primes, counts, best_score, deadline, max_check=100000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(primes)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(200):
            if time.time() >= deadline:
                break
            i1, i2 = rng.choice(n, size=2, replace=False)
            p1, p2 = primes[i1], primes[i2]
            r1, r2 = residues[i1], residues[i2]
            # try a few random pairs
            for _ in range(8):
                new_r1 = int(rng.integers(0, p1))
                new_r2 = int(rng.integers(0, p2))
                if new_r1 == r1 and new_r2 == r2:
                    continue
                os1 = r1 if r1 > 0 else p1
                os2 = r2 if r2 > 0 else p2
                ns1 = new_r1 if new_r1 > 0 else p1
                ns2 = new_r2 if new_r2 > 0 else p2
                if os1 <= max_check: counts[os1::p1] -= 1
                if os2 <= max_check: counts[os2::p2] -= 1
                if ns1 <= max_check: counts[ns1::p1] += 1
                if ns2 <= max_check: counts[ns2::p2] += 1
                s = _fast_score(counts, best_score, max_check)
                if s > best_score:
                    residues[i1] = new_r1
                    residues[i2] = new_r2
                    best_score = s
                    improved = True
                    break
                else:
                    if ns1 <= max_check: counts[ns1::p1] -= 1
                    if ns2 <= max_check: counts[ns2::p2] -= 1
                    if os1 <= max_check: counts[os1::p1] += 1
                    if os2 <= max_check: counts[os2::p2] += 1
    return residues, best_score

def _make_seed(primes, mode):
    n = len(primes)
    if mode == 0:
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
    if mode == 1:
        return [i % p for i, p in enumerate(primes)]
    if mode == 2:
        return [p // 2 for p in primes]
    if mode == 3:
        return [p - 1 for p in primes]
    # mode 4: random
    return [int(np.random.randint(0, p)) for p in primes]

def construct():
    primes = _sieve_primes(1000)
    max_check = 100000
    deadline = time.time() + 145

    # Try several seeds, keep the best.
    global_best = None
    global_best_score = -1
    for mode in range(5):
        if time.time() >= deadline - 5:
            break
        seed = _make_seed(primes, mode)
        sub_deadline = min(deadline - 1, time.time() + 25)
        res, sc = _full_search(seed, primes, sub_deadline, max_check)
        # multi-residue polish
        counts = _compute_counts(res, primes, max_check)
        res, sc = _multi_residue_moves(res, primes, counts, sc, deadline - 1, max_check)
        if sc > global_best_score:
            global_best_score = sc
            global_best = res

    if global_best is None:
        global_best = _make_seed(primes, 0)
        global_best_score = _score(global_best, primes)
    return global_best
# EVOLVE-BLOCK-END
