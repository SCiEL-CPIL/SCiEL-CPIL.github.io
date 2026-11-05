import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s hard kill

    N = 256
    elements = list(range(1, N + 1))

    # Build conflict graph as Python int bitmasks
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)

    # Precompute: for each vertex, the bitmask of vertices that CAN be added alongside it
    all_mask = ((1 << (N + 1)) - 1) & ~1  # bits 1..256 set
    non_conflict = [0] * (N + 1)
    for e in range(1, N + 1):
        non_conflict[e] = all_mask & ~(1 << e) & ~conflict[e]

    def popcount(x):
        return x.bit_count()

    def can_add(e, mask):
        return (conflict[e] & mask) == 0

    def fast_local_search(mask):
        """Efficient local search: repeatedly add all vertices that fit."""
        improved = True
        while improved:
            improved = False
            for e in range(1, N + 1):
                if not (mask & (1 << e)) and can_add(e, mask):
                    mask |= (1 << e)
                    improved = True
        return mask

    def greedy_build(order, start_mask=0):
        result = start_mask
        for e in order:
            if can_add(e, result):
                result |= (1 << e)
        return result

    def sa_improve(mask, max_time, rng_seed=42):
        """Simulated annealing with bitmask operations for speed."""
        current_mask = mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size

        temperature = 5.0
        cooling = 0.999
        rng = random.Random(rng_seed)

        while time.time() < max_time and temperature > 0.001:
            in_set = [e for e in range(1, N + 1) if current_mask & (1 << e)]
            not_in_set = [e for e in range(1, N + 1) if not (current_mask & (1 << e))]

            move_type = rng.random()

            if move_type < 0.4 and not_in_set:
                e = rng.choice(not_in_set)
                if can_add(e, current_mask):
                    new_mask = current_mask | (1 << e)
                    new_size = current_size + 1
                    current_mask = new_mask
                    current_size = new_size
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size

            elif move_type < 0.7 and current_size >= 2:
                e = rng.choice(in_set)
                candidate = current_mask & ~(1 << e)
                addable = []
                for f in range(1, N + 1):
                    if not (candidate & (1 << f)) and can_add(f, candidate):
                        addable.append(f)
                if len(addable) >= 2:
                    f, g = rng.sample(addable, 2)
                    if can_add(g, candidate | (1 << f)):
                        new_mask = candidate | (1 << f) | (1 << g)
                        new_size = current_size + 1
                        if new_size > current_size or rng.random() < math.exp((new_size - current_size) / temperature):
                            current_mask = new_mask
                            current_size = new_size
                            if current_size > best_size:
                                best_mask = current_mask
                                best_size = current_size

            else:
                if current_size >= 1 and not_in_set:
                    e = rng.choice(in_set)
                    candidate = current_mask & ~(1 << e)
                    addable = [f for f in range(1, N + 1) if not (candidate & (1 << f)) and can_add(f, candidate)]
                    if addable:
                        f = rng.choice(addable)
                        new_mask = candidate | (1 << f)
                        if rng.random() < 0.5:
                            current_mask = new_mask
                            current_size = current_size
                            if current_size > best_size:
                                best_mask = current_mask
                                best_size = current_size

            temperature *= cooling

        current_mask = fast_local_search(current_mask)
        current_size = popcount(current_mask)
        if current_size > best_size:
            best_mask = current_mask
            best_size = current_size

        return best_mask

    # Strategy 1: Start with all odds, greedily add evens by conflict count
    odds = [e for e in range(1, N + 1) if e % 2 == 1]
    evens = [e for e in range(1, N + 1) if e % 2 == 0]

    odds_mask = 0
    for e in odds:
        odds_mask |= (1 << e)

    evens_sorted = sorted(evens, key=lambda e: popcount(conflict[e]))
    mask = greedy_build(evens_sorted, odds_mask)
    mask = fast_local_search(mask)

    best_mask = mask
    best_size = popcount(mask)

    # Strategy 2: Randomized greedy restarts
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 30.0:
            break
        shuffled = evens[:]
        rng.shuffle(shuffled)
        m = greedy_build(shuffled, odds_mask)
        m = fast_local_search(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

    # Strategy 3: SA on best solution
    if time.time() < deadline - 15.0:
        sa_result = sa_improve(best_mask, deadline - 5.0, rng_seed=123)
        sa_size = popcount(sa_result)
        if sa_size > best_size:
            best_mask = sa_result
            best_size = sa_size

    # Strategy 4: More SA with different seeds
    if time.time() < deadline - 10.0:
        for seed in [456, 789]:
            if time.time() > deadline - 5.0:
                break
            sa_result = sa_improve(best_mask, deadline - 3.0, rng_seed=seed)
            sa_size = popcount(sa_result)
            if sa_size > best_size:
                best_mask = sa_result
                best_size = sa_size

    # Final polish
    if time.time() < deadline - 2.0:
        polished = fast_local_search(best_mask)
        if popcount(polished) > best_size:
            best_mask = polished
            best_size = popcount(polished)

    # Convert to list
    result = [e for e in range(1, N + 1) if best_mask & (1 << e)]

    if not result:
        result = [1]

    # Final validation
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                result.pop(j)
                j -= 1

    return result if result else [1]
