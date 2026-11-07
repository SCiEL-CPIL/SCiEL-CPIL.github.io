"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # We search over numbers k that are products of small primes with bounded exponents.
    # For k = prod p_i^{a_i}, k*sigma(k) = prod p_i^{a_i} * (p_i^{a_i+1}-1)/(p_i-1)
    # We want to find M values that appear most frequently.

    # Primes to use and max exponents (bounded to keep k*sigma(k) <= 10^18)
    primes = [2, 3, 5, 7, 11, 13]
    # For each prime p and exponent a, compute p^a * (p^(a+1)-1)/(p-1)
    # We need the product of these over all primes to be <= 10^18

    # Precompute for each prime the list of (exponent, factor) where factor = p^a * sigma(p^a)
    # sigma(p^a) = (p^(a+1)-1)/(p-1)
    # factor = p^a * (p^(a+1)-1)/(p-1)

    def compute_factors(p, max_exp):
        """Compute p^a * sigma(p^a) for a = 0, 1, ..., max_exp"""
        factors = []
        p_pow = 1  # p^0
        for a in range(max_exp + 1):
            if a == 0:
                factor = 1  # p^0 * sigma(p^0) = 1 * 1 = 1
            else:
                # p^a * (p^(a+1)-1)/(p-1)
                # = p^a * (1 + p + p^2 + ... + p^a)
                sigma_val = 0
                pp = 1
                for i in range(a + 1):
                    sigma_val += pp
                    pp *= p
                factor = p_pow * sigma_val
            factors.append((a, factor))
            p_pow *= p
        return factors

    # Determine max exponents: p^a * (p^(a+1)-1)/(p-1) <= 10^18
    # Roughly p^(2a+1) <= 10^18, so a <= 18*log(10)/log(p) - 0.5
    max_exps = []
    for p in primes:
        max_e = int(18 * math.log(10) / math.log(p)) + 2
        max_exps.append(max_e)

    # Build factor lists
    factor_lists = []
    for p, max_e in zip(primes, max_exps):
        fl = compute_factors(p, max_e)
        factor_lists.append(fl)

    # Now do a DFS to enumerate all combinations
    # For each combination of exponents (a0, a1, ..., a5), compute M = prod factors
    # Group by M, count solutions

    # To be efficient, we do DFS and prune when product > 10^18
    LIMIT = 10**18
    best_M = 1
    best_count = 0
    best_solutions = []

    # We'll collect all (M, k) pairs
    # k = prod p_i^{a_i}
    # M = prod factor_i

    # Use iterative DFS
    # State: index into primes, current product M, current k
    solutions_by_M = defaultdict(list)

    # To avoid excessive memory, we'll use a recursive approach with pruning
    def dfs(idx, current_M, current_k):
        nonlocal best_M, best_count, best_solutions
        if time.time() > deadline:
            return

        if idx == len(primes):
            if 1 <= current_M <= LIMIT:
                solutions_by_M[current_M].append(current_k)
            return

        p = primes[idx]
        for a, factor in factor_lists[idx]:
            new_M = current_M * factor
            if new_M > LIMIT:
                break
            new_k = current_k * (p ** a) if a > 0 else current_k
            dfs(idx + 1, new_M, new_k)

    dfs(0, 1, 1)

    # Find the M with the most solutions
    for M, ks in solutions_by_M.items():
        # Verify all k are distinct and k <= sqrt(M)
        valid_ks = []
        sqrt_M = math.isqrt(M)
        for k in ks:
            if k <= sqrt_M:
                valid_ks.append(k)
        # Remove duplicates
        valid_ks = list(set(valid_ks))
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_solutions = valid_ks

    # Verify the best solutions
    # sigma function for verification
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                result *= (d ** (exp + 1) - 1) // (d - 1)
            d += 1
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    # Verify each solution
    verified = []
    for k in best_solutions:
        if k * sigma(k) == best_M and k <= math.isqrt(best_M):
            verified.append(k)

    if len(verified) > 0:
        return {'M': int(best_M), 'k': [int(k) for k in verified]}
    else:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
