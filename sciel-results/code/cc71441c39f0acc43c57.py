"""Combined: prime-set selection + numpy coverage + SA local search on residues."""
import time
import random
import math
import numpy as np

_DEADLINE = 140.0
_L_CAP = 100000

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    global _START
    _START = time.time()

    primes = _primes_upto(1000)

    # Generate candidate prime sets with sum(1/p) <= 1
    def greedy_set(skip_set=None, max_p=None):
        s = []
        rem = 1.0
        for p in primes:
            if skip_set and p in skip_set:
                continue
            if max_p is not None and p > max_p:
                break
            if 1.0 / p <= rem + 1e-9:
                s.append(p)
                rem -= 1.0 / p
        return s

    sets_to_try = []

    # Set 1: primes >= 13 (seed-like, sum ~0.92)
    s1 = [p for p in primes if p >= 13]
    while s1 and sum(1.0 / p for p in s1) > 1.0:
        s1.pop()
    sets_to_try.append(s1)

    # Set 2: all small primes greedily
    sets_to_try.append(greedy_set())

    # Set 3: skip 2
    sets_to_try.append(greedy_set(skip_set={2}))

    # Set 4: skip 2, 3
    sets_to_try.append(greedy_set(skip_set={2, 3}))

    # Set 5: skip 2, 5
    sets_to_try.append(greedy_set(skip_set={2, 5}))

    # Set 6: skip 3, 5
    sets_to_try.append(greedy_set(skip_set={3, 5}))

    # Deduplicate
    unique_sets = []
    seen = set()
    for s in sets_to_try:
        if not s:
            continue
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)

    def eval_L(p_set, res_list):
        """Evaluate covered prefix length using numpy."""
        covered = np.zeros(_L_CAP + 1, dtype=bool)
        for i, p in enumerate(p_set):
            r = res_list[i]
            if r <= _L_CAP:
                covered[r::p] = True
        # Find first uncovered in 1..L_CAP
        not_covered = ~covered[1:]
        if not np.any(not_covered):
            return _L_CAP
        first_gap = int(np.argmax(not_covered))
        return first_gap

    def local_search(p_set, res_list, time_budget):
        """Hill-climbing on residues with occasional SA acceptance."""
        k = len(p_set)
        current_L = eval_L(p_set, res_list)
        best_L = current_L
        best_res = res_list[:]
        temp = 100.0
        cool = 0.995
        deadline = time.time() + time_budget

        while time.time() < deadline and temp > 0.1:
            i = random.randint(0, k - 1)
            p = p_set[i]
            old_r = res_list[i]

            if random.random() < 0.4:
                new_r = (old_r + 1) % p
            elif random.random() < 0.5:
                new_r = (old_r - 1) % p
            else:
                new_r = random.randint(0, p - 1)

            if new_r == old_r:
                continue

            res_list[i] = new_r
            new_L = eval_L(p_set, res_list)

            d = new_L - current_L
            if d >= 0 or random.random() < math.exp(d / max(temp, 0.01)):
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_res = res_list[:]
                    if best_L >= _L_CAP:
                        break
            else:
                res_list[i] = old_r

            temp *= cool

        return best_res, best_L

    def prime_swap_search(p_set, res_list, current_L, time_budget):
        """Try swapping primes in the set to improve coverage."""
        deadline = time.time() + time_budget
        best_L = current_L
        best_p = p_set[:]
        best_res = res_list[:]

        k = len(p_set)
        used = set(p_set)

        while time.time() < deadline:
            # Pick a random position
            i = random.randint(0, k - 1)
            old_p = p_set[i]
            # Pick a new prime not in the set
            candidates = [p for p in primes if p not in used and p != old_p]
            if not candidates:
                break
            new_p = random.choice(candidates)

            # Check budget
            old_budget = sum(1.0 / p for p in p_set)
            new_budget = old_budget - 1.0 / old_p + 1.0 / new_p
            if new_budget > 1.0 + 1e-9:
                continue

            p_set[i] = new_p
            used.discard(old_p)
            used.add(new_p)
            res_list[i] = random.randint(0, new_p - 1)

            new_L = eval_L(p_set, res_list)

            if new_L >= best_L:
                # Accept and do a quick local search
                res_list2, L2 = local_search(p_set, res_list, 2.0)
                if L2 > best_L:
                    best_L = L2
                    best_p = p_set[:]
                    best_res = res_list2[:]
            else:
                # Revert
                p_set[i] = old_p
                used.discard(new_p)
                used.add(old_p)

        return best_p, best_res, best_L

    best_L_global = 0
    best_p_global = None
    best_res_global = None

    n_sets = len(unique_sets)
    time_per_set = 130.0 / n_sets

    for p_set in unique_sets:
        if time.time() > _START + _DEADLINE - 5:
            break

        k = len(p_set)
        # Initialize residues: try to cover small numbers
        res_list = [0] * k
        # Greedy init: for each prime, pick residue that covers the most uncovered small numbers
        for i in range(k):
            p = p_set[i]
            best_r = 0
            best_count = 0
            for r in range(min(p, 50)):  # only try small residues for init
                # Count how many of 1..100 are covered by this residue
                cnt = sum(1 for x in range(1, 101) if x % p == r)
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            res_list[i] = best_r

        # Local search
        res_list, L = local_search(p_set, res_list, time_per_set * 0.7)
        if L > best_L_global:
            best_L_global = L
            best_p_global = p_set[:]
            best_res_global = res_list[:]

        # Prime swap search
        if time.time() < _START + _DEADLINE - 10:
            p_set2, res_list2, L2 = prime_swap_search(p_set, res_list, L, time_per_set * 0.3)
            if L2 > best_L_global:
                best_L_global = L2
                best_p_global = p_set2[:]
                best_res_global = res_list2[:]

        if best_L_global >= _L_CAP:
            break

    # Final polish on best
    if best_p_global is not None and time.time() < _START + _DEADLINE - 3:
        best_res_global, best_L_final = local_search(best_p_global, best_res_global, 5.0)
        if best_L_final > best_L_global:
            best_L_global = best_L_final

    if best_p_global is None:
        # Fallback
        p_set = [p for p in primes if p >= 13]
        while p_set and sum(1.0 / p for p in p_set) > 1.0:
            p_set.pop()
        res_list = [(i + 1) % p for i, p in enumerate(p_set)]
        return [[p, r] for p, r in zip(p_set, res_list)]

    # Validate budget with exact fractions
    from fractions import Fraction
    budget = sum(Fraction(1, p) for p in best_p_global)
    if budget > 1:
        # Remove primes with smallest p (largest 1/p) until valid
        idx = sorted(range(len(best_p_global)), key=lambda i: best_p_global[i])
        while budget > 1 and best_p_global:
            i = idx.pop(0)
            budget -= Fraction(1, best_p_global[i])
            del best_p_global[i]
            del best_res_global[i]

    return [[p, r] for p, r in zip(best_p_global, best_res_global)]
