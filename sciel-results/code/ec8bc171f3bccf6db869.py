import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82
    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            v = _lcm(i, j)
            lcm_table[i][j] = v
            lcm_table[j][i] = v

    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    num_bad = len(bad_triples)

    # For each element x, list of (y, z) pairs such that (x,y,z) is a bad triple
    x_pairs = [[] for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        x_pairs[a].append((b, c))
        x_pairs[b].append((a, c))
        x_pairs[c].append((a, b))

    # Bitmask helpers: element x (1..128) -> bit (x-1)
    def bit(x):
        return 1 << (x - 1)

    def has(mask, x):
        return (mask >> (x - 1)) & 1

    def set_bit(mask, x):
        return mask | bit(x)

    def clear_bit(mask, x):
        return mask & ~bit(x)

    def popcount(m):
        return bin(m).count('1')

    # Count violations if we add element x to mask
    def add_violations(mask, x):
        cnt = 0
        for (y, z) in x_pairs[x]:
            if has(mask, y) and has(mask, z):
                cnt += 1
        return cnt

    # Count total violations in mask (for SA energy)
    def total_violations(mask):
        cnt = 0
        for (a, b, c) in bad_triples:
            if has(mask, a) and has(mask, b) and has(mask, c):
                cnt += 1
        return cnt

    # Greedy construction: start empty, add elements that don't create violations
    # Try multiple orderings
    def greedy_construct(order):
        mask = 0
        for x in order:
            if add_violations(mask, x) == 0:
                mask = set_bit(mask, x)
        return mask

    # Local search: 1-swap (remove one, add one or more)
    def local_search(mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r in range(1, n + 1):
                if not has(mask, r):
                    continue
                new_mask = clear_bit(mask, r)
                # Try adding all possible elements
                added = 0
                for x in range(1, n + 1):
                    if not has(new_mask, x) and add_violations(new_mask, x) == 0:
                        new_mask = set_bit(new_mask, x)
                        added += 1
                if added > 1:
                    mask = new_mask
                    improved = True
        return mask

    # 2-swap local search: remove two, add up to four
    def two_swap_search(mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r1 in range(1, n + 1):
                if not has(mask, r1):
                    continue
                for r2 in range(r1 + 1, n + 1):
                    if not has(mask, r2):
                        continue
                    if time.time() > time_limit:
                        break
                    new_mask = clear_bit(clear_bit(mask, r1), r2)
                    added = 0
                    for x in range(1, n + 1):
                        if not has(new_mask, x) and add_violations(new_mask, x) == 0:
                            new_mask = set_bit(new_mask, x)
                            added += 1
                    if added > 2:
                        mask = new_mask
                        improved = True
        return mask

    # Simulated annealing
    def simulated_annealing(initial_mask, time_limit, seed):
        import random
        rng = random.Random(seed)
        mask = initial_mask
        cur_energy = total_violations(mask)
        best_mask = mask
        best_size = popcount(mask)
        T = 10.0
        cool = 0.9995

        while time.time() < time_limit and T > 0.01:
            # Pick a random move: toggle a random element
            x = rng.randint(1, n)
            new_mask = set_bit(mask, x) if not has(mask, x) else clear_bit(mask, x)
            new_energy = total_violations(new_mask)
            delta = new_energy - cur_energy
            new_size = popcount(new_mask)

            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                mask = new_mask
                cur_energy = new_energy
                if new_size > best_size:
                    best_size = new_size
                    best_mask = mask
            T *= cool

        return best_mask

    # Phase 1: Multiple greedy constructions with different orderings
    best_mask = 0
    best_size = 0

    # Natural order
    m = greedy_construct(list(range(1, n + 1)))
    s = popcount(m)
    if s > best_size:
        best_size = s
        best_mask = m

    # Reverse order
    m = greedy_construct(list(range(n, 0, -1)))
    s = popcount(m)
    if s > best_size:
        best_size = s
        best_mask = m

    # Sort by number of conflicts (fewest first)
    conflict_count = [len(x_pairs[x]) for x in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: conflict_count[x])
    m = greedy_construct(order)
    s = popcount(m)
    if s > best_size:
        best_size = s
        best_mask = m

    # Random orderings
    import random
    for seed in range(20):
        if time.time() > deadline - 40:
            break
        rng = random.Random(seed * 12345 + 678)
        order = list(range(1, n + 1))
        rng.shuffle(order)
        m = greedy_construct(order)
        s = popcount(m)
        if s > best_size:
            best_size = s
            best_mask = m

    # Phase 2: Local search on best
    best_mask = local_search(best_mask, deadline - 35)
    best_size = popcount(best_mask)

    # Phase 3: 2-swap search
    best_mask = two_swap_search(best_mask, deadline - 25)
    best_size = popcount(best_mask)

    # Phase 4: More local search
    best_mask = local_search(best_mask, deadline - 20)
    best_size = popcount(best_mask)

    # Phase 5: Simulated annealing from best
    best_mask = simulated_annealing(best_mask, deadline - 10, 42)
    best_size = popcount(best_mask)

    # Phase 6: Final local search
    best_mask = local_search(best_mask, deadline - 5)
    best_size = popcount(best_mask)

    # Phase 7: Try SA from a few other starting points
    for seed in range(5):
        if time.time() > deadline - 5:
            break
        m = greedy_construct(list(range(1, n + 1)))
        # Perturb
        import random
        rng = random.Random(seed * 999 + 1)
        for _ in range(rng.randint(5, 20)):
            x = rng.randint(1, n)
            m = set_bit(m, x) if not has(m, x) else clear_bit(m, x)
        # Fix violations
        for _ in range(100):
            changed = False
            for (a, b, c) in bad_triples:
                if has(m, a) and has(m, b) and has(m, c):
                    # Remove the one with most conflicts
                    best_x = a
                    best_conf = 0
                    for x in [a, b, c]:
                        conf = add_violations(m, x)
                        if conf > best_conf:
                            best_conf = conf
                            best_x = x
                    m = clear_bit(m, best_x)
                    changed = True
            if not changed:
                break
        # SA
        m = simulated_annealing(m, deadline - 5, seed + 100)
        s = popcount(m)
        if s > best_size:
            best_size = s
            best_mask = m

    # Final polish: add any element with 0 violations
    for x in range(1, n + 1):
        if not has(best_mask, x) and add_violations(best_mask, x) == 0:
            best_mask = set_bit(best_mask, x)

    # Verify
    for (a, b, c) in bad_triples:
        if has(best_mask, a) and has(best_mask, b) and has(best_mask, c):
            # Should not happen; remove one
            best_mask = clear_bit(best_mask, c)

    result = [x for x in range(1, n + 1) if has(best_mask, x)]
    if not result:
        result = [1]

    return result
