"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _sieve_primes(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill

    n = 256

    # --- Candidate 1: primes + 1 (guaranteed valid) ---
    primes = _sieve_primes(n)
    prime_set = [1] + primes
    prime_sum = sum(prime_set)

    # --- Candidate 2: greedy descending (current best approach) ---
    selected = [1]
    for a in range(n, 1, -1):
        if time.time() - start_time > deadline:
            break
        can_add = True
        for b in selected:
            if gcd(a, b) != 1:
                can_add = False
                break
        if can_add:
            selected.append(a)
    greedy_sum = sum(selected)

    # Pick the better starting point
    best = selected if greedy_sum >= prime_sum else prime_set
    best_sum = max(greedy_sum, prime_sum)
    best_set = list(best)
    best_set = sorted(best_set)  # normalize

    # --- Local search: try to improve by adding or swapping ---
    # Build set for fast membership
    best_in_set = set(best_set)

    # Precompute: for each number 2..256, the set of numbers in 1..256 it's coprime with
    # Actually, we'll just check on the fly with gcd, but limit the search.

    # Try to add any number not in best_set that is coprime to all in best_set
    # Process in descending order to prefer larger additions
    for a in range(n, 1, -1):
        if time.time() - start_time > deadline:
            break
        if a in best_in_set:
            continue
        can_add = True
        for b in best_set:
            if gcd(a, b) != 1:
                can_add = False
                break
        if can_add:
            best_set.append(a)
            best_in_set.add(a)
            best_sum += a

    # Now try swaps: for each element in best_set (excluding 1), see if we can
    # replace it with a larger number not in best_set that is coprime to all others.
    # To keep it fast, limit the number of swap passes.
    max_swap_passes = 3
    for _pass in range(max_swap_passes):
        if time.time() - start_time > deadline:
            break
        improved = False
        # Sort best_set descending to try removing smaller elements first for larger swaps
        for i, b in enumerate(sorted(best_set, reverse=True)):
            if time.time() - start_time > deadline:
                break
            if b == 1:
                continue
            # Try to find a larger number a > b not in best_set
            # that is coprime to all elements in best_set except b
            others = [x for x in best_set if x != b]
            for a in range(n, b, -1):
                if a in best_in_set:
                    continue
                can_add = True
                for c in others:
                    if gcd(a, c) != 1:
                        can_add = False
                        break
                if can_add:
                    # Perform swap
                    best_set.remove(b)
                    best_in_set.discard(b)
                    best_set.append(a)
                    best_in_set.add(a)
                    best_sum += a - b
                    improved = True
                    break
        if not improved:
            break

    # Ensure return type is list of Python ints
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
