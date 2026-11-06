"""Maximize coprime adjacent divisor pairs for n <= 10^18."""
import math
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
_MAX_VAL = 10**18
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
_NUM_PRIMES = 20  # primes to consider in DFS


def _tau_perp_from_factors(factors):
    """Compute tau_perp from a list of (prime, exponent) pairs."""
    if not factors:
        return 0
    # Generate all divisors
    divs = [1]
    for p, e in factors:
        cur_len = len(divs)
        factor = 1
        for _ in range(e):
            factor *= p
            for i in range(cur_len):
                divs.append(divs[i] * factor)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count


def _factors_from_exponents(primes_used, exponents):
    """Build factor list from primes and exponents."""
    return [(primes_used[i], exponents[i]) for i in range(len(primes_used)) if exponents[i] > 0]


def _dfs_generate(start_time, gen_deadline):
    """Generate candidates with non-increasing exponents using DFS."""
    candidates = []  # (tau, n, exponents_tuple)

    def dfs(idx, max_exp, current_n, current_tau, exponents):
        if time.time() > gen_deadline:
            return
        if current_n > 1:
            candidates.append((current_tau, current_n, tuple(exponents[:idx])))

        if idx >= _NUM_PRIMES:
            return

        p = _PRIMES[idx]
        limit_n = _MAX_VAL // current_n

        # Max exponent for this prime, constrained by max_exp and value limit
        e_max = 0
        temp = 1
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1

        for e in range(1, e_max + 1):
            exponents.append(e)
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau, exponents)
            exponents.pop()

            if time.time() > gen_deadline:
                return

    dfs(0, 60, 1, 1, [])
    return candidates


def _local_search(best_n, best_score, start_time, eval_deadline):
    """Try small perturbations around best_n."""
    # Factorize best_n
    factors = []
    temp = best_n
    for p in _PRIMES:
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            factors.append((p, e))
        if temp == 1:
            break
    if temp > 1:
        factors.append((temp, 1))

    best_n_local = best_n
    best_score_local = best_score

    # Try increasing/decreasing exponents by 1
    for i, (p, e) in enumerate(factors):
        if time.time() > eval_deadline:
            break
        # Try e+1
        new_factors = factors.copy()
        new_factors[i] = (p, e + 1)
        n_test = 1
        for pp, ee in new_factors:
            n_test *= pp ** ee
        if n_test <= _MAX_VAL:
            score_test = _tau_perp_from_factors(new_factors)
            if score_test > best_score_local:
                best_score_local = score_test
                best_n_local = n_test

        # Try e-1
        if e > 1:
            new_factors = factors.copy()
            new_factors[i] = (p, e - 1)
            n_test = 1
            for pp, ee in new_factors:
                n_test *= pp ** ee
            if n_test >= 2 and n_test <= _MAX_VAL:
                score_test = _tau_perp_from_factors(new_factors)
                if score_test > best_score_local:
                    best_score_local = score_test
                    best_n_local = n_test

    # Try replacing a prime with a nearby prime
    for i, (p, e) in enumerate(factors):
        if time.time() > eval_deadline:
            break
        for j, q in enumerate(_PRIMES):
            if q == p:
                continue
            # Only try primes close in index to avoid huge changes
            if abs(j - i) > 3:
                continue
            new_factors = factors.copy()
            new_factors[i] = (q, e)
            n_test = 1
            for pp, ee in new_factors:
                n_test *= pp ** ee
            if 2 <= n_test <= _MAX_VAL:
                score_test = _tau_perp_from_factors(new_factors)
                if score_test > best_score_local:
                    best_score_local = score_test
                    best_n_local = n_test

    return best_n_local, best_score_local


def construct():
    start_time = time.time()
    total_deadline = start_time + 140  # safety margin for 150s limit
    gen_deadline = start_time + 60     # spend up to 60s generating candidates

    # Generate candidates via DFS
    candidates = _dfs_generate(start_time, gen_deadline)

    if not candidates:
        # Fallback: primorial
        n = 1
        for p in _PRIMES[:15]:
            n *= p
        return n

    # Sort by tau descending (proxy for potential tau_perp)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Evaluate top candidates
    eval_deadline = total_deadline - 10  # leave 10s for local search
    best_score = -1
    best_n = candidates[0][1]
    eval_count = 0
    max_evals = 5000  # cap evaluations to stay within time

    for tau, n, exps in candidates:
        if time.time() > eval_deadline:
            break
        if eval_count >= max_evals:
            break
        score = _tau_perp_from_factors([(p, e) for p, e in zip(_PRIMES, exps) if e > 0])
        if score > best_score:
            best_score = score
            best_n = n
        eval_count += 1

    # Local search around the best
    if time.time() < total_deadline - 5:
        best_n, best_score = _local_search(best_n, best_score, start_time, total_deadline - 5)

    return best_n
# EVOLVE-BLOCK-END
