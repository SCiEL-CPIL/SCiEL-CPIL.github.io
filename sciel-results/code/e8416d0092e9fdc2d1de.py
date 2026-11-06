"""CSP-style search with prime-exponent vectors, forward checking, and M-iteration."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    timeout = 55.0  # safety margin before 90s kill

    target = factorial(12)  # 479001600

    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    def prime_factorization(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    target_pf = prime_factorization(target)
    primes = sorted(target_pf.keys())
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)

    def pf_to_vec(pf):
        vec = [0] * n_primes
        for p, e in pf.items():
            vec[prime_to_idx[p]] = e
        return tuple(vec)

    target_vec = pf_to_vec(target_pf)

    def vec_to_int(vec):
        """Convert exponent vector back to integer (for validation)."""
        result = 1
        for i, p in enumerate(primes):
            if vec[i] > 0:
                result *= p ** vec[i]
        return result

    # Precompute: for each divisor d > 12 of target, store its exponent vector
    def get_divisors(n):
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)

    all_divs = get_divisors(target)
    valid_divs = [d for d in all_divs if d > 12]

    # Map each valid divisor to its exponent vector
    divisor_vecs = {}
    for d in valid_divs:
        divisor_vecs[d] = pf_to_vec(prime_factorization(d))

    def can_divide(rem_vec, div_vec):
        """Check if div_vec divides rem_vec (component-wise)."""
        for i in range(n_primes):
            if rem_vec[i] < div_vec[i]:
                return False
        return True

    def sub_vec(rem_vec, div_vec):
        """Subtract div_vec from rem_vec."""
        return tuple(rem_vec[i] - div_vec[i] for i in range(n_primes))

    def is_zero_vec(vec):
        return all(v == 0 for v in vec)

    # Strategy: iterate M (max allowed factor) in increasing order.
    # For each M, try to find a valid decomposition using only factors <= M.
    # The first M for which we find a solution gives the optimal answer.

    best_solution = None
    best_max = float('inf')

    n_valid = len(valid_divs)

    # Precompute a lookup: for each remaining vector, which candidate indices
    # can divide it. We'll build this on the fly per M.

    for M_idx in range(n_valid):
        if time.time() - start_time > timeout:
            break

        M = valid_divs[M_idx]

        # Candidates are valid_divs[0 ... M_idx]
        candidates = valid_divs[:M_idx + 1]
        candidate_vecs = [divisor_vecs[c] for c in candidates]
        n_cand = len(candidates)

        found = None

        def dfs(start_idx, rem_vec, current_factors):
            nonlocal found
            if time.time() - start_time > timeout:
                return

            if is_zero_vec(rem_vec):
                if len(current_factors) >= 1:
                    found = list(current_factors)
                return

            if found is not None:
                return

            if len(current_factors) >= 12:
                return

            # Forward checking: for each prime, the remaining exponent must be
            # coverable by some subset of the remaining candidates.
            # Quick necessary condition: the sum of exponents for each prime
            # across all remaining candidates (from start_idx onward) must be >= rem_vec.
            # We can precompute suffix sums for each prime.

            # For now, do a lighter check: for each prime i where rem_vec[i] > 0,
            # there must be at least one candidate from start_idx onward that has
            # that prime in its factorization.
            for i in range(n_primes):
                if rem_vec[i] > 0:
                    found_any = False
                    for j in range(start_idx, n_cand):
                        if candidate_vecs[j][i] > 0:
                            found_any = True
                            break
                    if not found_any:
                        return

            for i in range(start_idx, n_cand):
                if found is not None:
                    return

                c = candidates[i]
                cv = candidate_vecs[i]

                # c must divide rem_vec
                if not can_divide(rem_vec, cv):
                    continue

                new_rem = sub_vec(rem_vec, cv)

                current_factors.append(c)
                dfs(i + 1, new_rem, current_factors)
                current_factors.pop()

                if found is not None:
                    return

            return None

        dfs(0, target_vec, [])

        if found is not None:
            return found

    # Fallback: return [target]
    return [target]

# EVOLVE-BLOCK-END
