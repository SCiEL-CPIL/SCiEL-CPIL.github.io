"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    primes = _sieve(1000)
    max_check = 100000

    def compute_L(covered):
        """Compute the largest L such that 1..L are all covered."""
        sub = covered[1:]
        false_indices = np.where(~sub)[0]
        if len(false_indices) == 0:
            return max_check
        return false_indices[0]

    def greedy_residues_numpy(prime_list, max_check=100000):
        """Assign residues greedily using numpy for speed."""
        if not prime_list:
            return []

        covered = np.zeros(max_check + 1, dtype=bool)
        classes = []

        for p in prime_list:
            if time.time() > deadline:
                break

            # Find the first uncovered number
            sub = covered[1:]
            false_indices = np.where(~sub)[0]
            if len(false_indices) == 0:
                break  # All covered
            first_uncovered = int(false_indices[0])

            r_target = first_uncovered % p
            best_r = r_target
            best_L = first_uncovered - 1

            # For small primes, try all residues; for large, try target + neighbors
            if p <= 50:
                residues_to_try = list(range(p))
            else:
                residues_to_try = [r_target]
                for d in range(-5, 6):
                    r_alt = (r_target + d) % p
                    if r_alt != r_target:
                        residues_to_try.append(r_alt)

            for r in residues_to_try:
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                mask = np.zeros(max_check + 1, dtype=bool)
                mask[start::p] = True
                new_covered = covered | mask
                new_L = compute_L(new_covered)
                if new_L > best_L:
                    best_L = new_L
                    best_r = r

            classes.append([p, int(best_r)])

            # Update covered array
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            covered[start::p] = True

        return classes

    best_classes = []
    best_L = 0

    def evaluate_strategy(selected_primes):
        nonlocal best_classes, best_L
        if time.time() > deadline - 5:
            return
        if not selected_primes:
            return
        classes = greedy_residues_numpy(selected_primes, max_check)
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            covered[start::p] = True
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = classes

    # Strategy 1: smallest primes first (densest)
    primes_asc = sorted(primes)
    budget = Fraction(1)
    selected = []
    for p in primes_asc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected)

    # Strategy 2: largest primes first (cheapest, most primes)
    primes_desc = sorted(primes, reverse=True)
    budget = Fraction(1)
    selected = []
    for p in primes_desc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected)

    # Strategy 3: mixed - use small primes for density, fill with large
    if time.time() < deadline - 10:
        for num_small in range(1, 8):
            if time.time() > deadline - 10:
                break
            small_primes = primes_asc[:num_small]
            budget = Fraction(1)
            for p in small_primes:
                budget -= Fraction(1, p)
            if budget < 0:
                break

            selected = list(small_primes)
            for p in primes_desc:
                if p in small_primes:
                    continue
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)

            evaluate_strategy(selected)

    # Strategy 4: skip some small primes to fit more medium primes
    if time.time() < deadline - 10:
        for exclude in [[2], [3], [2,3], [5], [2,5]]:
            if time.time() > deadline - 10:
                break
            budget = Fraction(1)
            selected = []
            for p in primes_asc:
                if p in exclude:
                    continue
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
                else:
                    break
            evaluate_strategy(selected)

    return best_classes if best_classes else [[2, 1]]

# EVOLVE-BLOCK-END
