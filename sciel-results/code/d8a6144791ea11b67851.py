import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(2000)

    def get_divisors_sorted(factors):
        """Generate all divisors of n given its prime factorization, sorted."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        divs.sort()
        return divs

    def tau_perp_fast(factors):
        """Compute tau_perp score from prime factorization."""
        divs = get_divisors_sorted(factors)
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    def compute_product(factors):
        prod = 1
        for p, e in factors:
            prod *= p ** e
        return prod

    # Try several initial configurations
    def try_initial_config(prime_list, exp_list):
        factors = list(zip(prime_list, exp_list))
        prod = compute_product(factors)
        if prod > 10**18 or prod < 2:
            return None, 0, []
        score = tau_perp_fast(factors)
        return prod, score, factors

    # Configuration 1: First 12 primes with decreasing exponents
    config1_primes = primes[:12]
    config1_exps = [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    n1, s1, f1 = try_initial_config(config1_primes, config1_exps)

    # Configuration 2: First 15 primes, all exponent 1 (squarefree)
    config2_primes = primes[:15]
    config2_exps = [1] * 15
    n2, s2, f2 = try_initial_config(config2_primes, config2_exps)

    # Configuration 3: First 10 primes with higher exponents on small primes
    config3_primes = primes[:10]
    config3_exps = [6, 3, 2, 1, 1, 1, 1, 1, 1, 1]
    n3, s3, f3 = try_initial_config(config3_primes, config3_exps)

    # Configuration 4: First 13 primes with moderate exponents
    config4_primes = primes[:13]
    config4_exps = [3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    n4, s4, f4 = try_initial_config(config4_primes, config4_exps)

    # Pick the best initial
    candidates = [(n1, s1, f1), (n2, s2, f2), (n3, s3, f3), (n4, s4, f4)]
    candidates = [(n, s, f) for n, s, f in candidates if n is not None and n >= 2]
    
    if not candidates:
        # Fallback: just use first few primes
        factors = [(p, 1) for p in primes[:10]]
        prod = compute_product(factors)
        if prod > 10**18:
            factors = [(p, 1) for p in primes[:8]]
            prod = compute_product(factors)
        score = tau_perp_fast(factors)
        return prod

    best_n, best_score, best_factors = max(candidates, key=lambda x: x[1])

    current_factors = list(best_factors)
    current_prod = best_n
    current_score = best_score

    # Local search: best-improvement hill climbing
    improved = True
    no_improve_count = 0
    max_no_improve = 3

    while improved and time.time() < deadline:
        improved = False
        best_move_score = current_score
        best_new_factors = None
        best_new_prod = current_prod

        current_primes_set = {p for p, e in current_factors}

        # Move 1: Change exponent of existing prime
        for i, (p, e) in enumerate(current_factors):
            # Try increasing exponent
            if current_prod * p <= 10**18:
                new_factors = list(current_factors)
                new_factors[i] = (p, e + 1)
                new_prod = current_prod * p
                s = tau_perp_fast(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

            # Try decreasing exponent
            if e > 1:
                new_factors = list(current_factors)
                new_factors[i] = (p, e - 1)
                new_prod = current_prod // p
                if new_prod >= 2:
                    s = tau_perp_fast(new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod

        # Move 2: Swap prime p with prime q (keeping same exponent)
        for i, (p, e) in enumerate(current_factors):
            for q in primes:
                if q in current_primes_set:
                    continue
                base = current_prod // (p ** e)
                new_prod = base * (q ** e)
                if new_prod > 10**18 or new_prod < 2:
                    continue
                new_factors = list(current_factors)
                new_factors[i] = (q, e)
                s = tau_perp_fast(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # Move 3: Add a new prime with exponent 1
        for q in primes:
            if q in current_primes_set:
                continue
            new_prod = current_prod * q
            if new_prod > 10**18:
                break
            new_factors = list(current_factors)
            new_factors.append((q, 1))
            s = tau_perp_fast(new_factors)
            if s > best_move_score:
                best_move_score = s
                best_new_factors = new_factors
                best_new_prod = new_prod

        # Move 4: Remove a prime (only if exponent is 1, or reduce exponent)
        for i, (p, e) in enumerate(current_factors):
            if e == 1 and len(current_factors) > 2:
                new_factors = list(current_factors)
                del new_factors[i]
                new_prod = current_prod // p
                if new_prod >= 2:
                    s = tau_perp_fast(new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod

        if best_new_factors is not None:
            current_factors = best_new_factors
            current_prod = best_new_prod
            current_score = best_move_score
            improved = True
            no_improve_count = 0

            if current_score > best_score:
                best_score = current_score
                best_n = current_prod
                best_factors = list(current_factors)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    # If time remains, try random perturbations
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() >= deadline - 2:
                break
            # Randomly perturb the current solution
            perturbed = list(current_factors)
            # Randomly change one exponent
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                p, e = perturbed[idx]
                delta = random.choice([-1, 1, 2, -2])
                new_e = e + delta
                if new_e >= 1:
                    perturbed[idx] = (p, new_e)
                    prod = compute_product(perturbed)
                    if 2 <= prod <= 10**18:
                        score = tau_perp_fast(perturbed)
                        if score > best_score:
                            best_score = score
                            best_n = prod
                            best_factors = list(perturbed)
                            current_factors = list(perturbed)
                            current_prod = prod
                            current_score = score

    return best_n
