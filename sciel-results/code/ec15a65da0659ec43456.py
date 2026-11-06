"""Maximize cardinality of a dissociated reciprocal subset of [1,22].

A set S is dissociated (all reciprocal subset sums distinct) iff the integer
subset sums of the weights w_i = L/(i+1) (L = lcm(1..22)) are all distinct.
We build via randomized greedy restarts + exact local search (swap/add),
keeping the largest valid set found before a wall-clock deadline.
"""
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before the 90 s kill

    N = 22
    L = reduce(_lcm, range(1, N + 1))
    # weight for element (i+1); element x -> weight L//x
    weights = [L // (i + 1) for i in range(N)]  # weights[i] for element i+1

    def time_up():
        return time.time() > deadline

    # sums for a mask: all subset sums of weights[i] for i in mask
    def mask_sums(mask):
        sums = [0]
        for i in range(N):
            if (mask >> i) & 1:
                w = weights[i]
                sums += [s + w for s in sums]
        return set(sums)

    def can_add(sums, w):
        """True if adding weight w to a set with subset sums `sums` keeps all distinct."""
        for s in sums:
            if s + w in sums:
                return False
        return True

    def greedy(order, base_mask=0):
        """Greedily add elements in `order` (0-based) to base_mask if no collision.
        Returns (mask, size)."""
        mask = base_mask
        sums = mask_sums(base_mask)
        size = bin(base_mask).count("1")
        for i in order:
            if (mask >> i) & 1:
                continue
            w = weights[i]
            if can_add(sums, w):
                sums = sums | {s + w for s in sums}
                mask |= (1 << i)
                size += 1
        return mask, size

    def local_search(mask, budget):
        """Try to improve mask via single swaps and adds. Returns best mask."""
        sums = mask_sums(mask)
        size = bin(mask).count("1")
        improved = True
        while improved and not time_up():
            improved = False
            # Try adding each unused element
            for j in range(N):
                if (mask >> j) & 1:
                    continue
                if can_add(sums, weights[j]):
                    mask |= (1 << j)
                    size += 1
                    sums = sums | {s + weights[j] for s in sums}
                    improved = True
                    break
            # Try single swaps: remove i (in), add j (out)
            for i in range(N):
                if not ((mask >> i) & 1):
                    continue
                for j in range(N):
                    if (mask >> j) & 1:
                        continue
                    # remove i, then test add j
                    new_mask = mask & ~(1 << i)
                    new_sums = mask_sums(new_mask)
                    if can_add(new_sums, weights[j]):
                        new_mask |= (1 << j)
                        # commit
                        mask = new_mask
                        sums = mask_sums(mask)
                        improved = True
                        break
                if improved:
                    break
        return mask

    best_mask = 0
    best_size = 0

    def consider(mask, size):
        nonlocal best_mask, best_size
        if size > best_size:
            best_mask = mask
            best_size = size

    # --- deterministic orderings ---
    orders = []
    orders.append(sorted(range(N), key=lambda i: weights[i], reverse=True))  # big weights first
    orders.append(sorted(range(N), key=lambda i: weights[i]))                 # small weights first
    orders.append(sorted(range(N)))                                           # ascending element
    orders.append(sorted(range(N), reverse=True))                             # descending element
    # small-then-large packing (fine elements first, then coarse)
    orders.append(sorted(range(N), key=lambda i: (weights[i] % (L // 2))))
    for od in orders:
        m, s = greedy(od)
        consider(m, s)
        if not time_up():
            m2, s2 = local_search(m, None)
            consider(m2, s2)

    # --- randomized restarts ---
    rng = random.Random(20240517)
    it = 0
    while not time_up():
        it += 1
        # mix of random and structured orderings
        if it % 5 == 0:
            od = sorted(range(N), key=lambda i: (weights[i], rng.random()))
        elif it % 5 == 1:
            od = sorted(range(N), key=lambda i: (weights[i] % (L // 3), rng.random()))
        else:
            od = list(range(N))
            rng.shuffle(od)
        m, s = greedy(od)
        consider(m, s)
        if s >= best_size and not time_up():
            m2, s2 = local_search(m, None)
            consider(m2, s2)
        # occasional local search from a random small seed
        if it % 13 == 0 and not time_up():
            seed_mask = 0
            for _ in range(rng.randint(2, 6)):
                seed_mask |= (1 << rng.randrange(N))
            m3, s3 = local_search(seed_mask, None)
            consider(m3, s3)

    # --- final polish on the best ---
    if best_mask and not time_up():
        m_final = local_search(best_mask, None)
        consider(m_final, bin(m_final).count("1"))

    if best_mask == 0:
        return [1]

    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        return [1]
    return result
