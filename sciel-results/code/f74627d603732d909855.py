"""Combined program: prime-power baseline with exact branch-and-bound for 2-way and 3-way merges."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # For each prime, find the largest prime power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Baseline: 1 + all prime powers
    baseline = [1] + [prime_powers[p] for p in primes]
    baseline_sum = sum(baseline)

    # Identify all possible merge options:
    # A merge option is a set of primes whose product <= N,
    # and the product gives more sum than the individual prime powers.
    # We consider 2-way and 3-way merges.

    merge_options = []  # (gain, frozenset_of_primes, product_value)

    # 2-way merges
    for i in range(len(primes)):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            Pq = prime_powers[q]
            prod = p * q
            if prod > N:
                break
            gain = prod - Pp - Pq
            if gain > 0:
                merge_options.append((gain, frozenset([p, q]), prod))

    # 3-way merges
    for i in range(len(primes)):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            Pq = prime_powers[q]
            pq = p * q
            if pq > N:
                break
            for k in range(j + 1, len(primes)):
                r = primes[k]
                Pr = prime_powers[r]
                pqr = pq * r
                if pqr > N:
                    break
                gain = pqr - Pp - Pq - Pr
                if gain > 0:
                    merge_options.append((gain, frozenset([p, q, r]), pqr))

    # Which primes can participate in merges?
    mergeable_primes = set()
    for gain, prime_set, prod in merge_options:
        mergeable_primes.update(prime_set)

    # Sort merge options by gain descending for better pruning
    merge_options.sort(key=lambda x: -x[0])

    # Use branch and bound to find optimal selection of non-overlapping merges
    best_gain = [0]
    best_selection = [[]]  # list of (gain, prime_set, prod)

    def dfs(idx, used_mask, current_gain, current_selection):
        if time.time() > deadline - 5:
            return
        if current_gain > best_gain[0]:
            best_gain[0] = current_gain
            best_selection[0] = list(current_selection)

        if idx >= len(merge_options):
            return

        # Pruning: optimistic upper bound
        remaining_max = sum(opt[0] for opt in merge_options[idx:])
        if current_gain + remaining_max <= best_gain[0]:
            return

        # Option 1: skip this merge option
        dfs(idx + 1, used_mask, current_gain, current_selection)

        # Option 2: take this merge option if no prime overlap
        gain, prime_set, prod = merge_options[idx]
        can_take = True
        for p in prime_set:
            if p in used_mask:
                can_take = False
                break
        if can_take:
            new_used = used_mask | prime_set
            current_selection.append((gain, prime_set, prod))
            dfs(idx + 1, new_used, current_gain + gain, current_selection)
            current_selection.pop()

    dfs(0, set(), 0, [])

    # Construct final result
    result = [1]
    used_primes = set()
    for gain, prime_set, prod in best_selection[0]:
        used_primes.update(prime_set)

    for p in primes:
        if p not in used_primes:
            result.append(prime_powers[p])

    # Add the merged products
    for gain, prime_set, prod in best_selection[0]:
        result.append(prod)

    # Verify all constraints
    if len(set(result)) != len(result):
        result = baseline

    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            result = baseline
            break

    # Check pairwise coprime
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        result = baseline

    # Ensure we return Python ints in a list
    result = [int(x) for x in result]

    # Deadline check
    if time.time() > deadline:
        pass

    return result

# EVOLVE-BLOCK-END
