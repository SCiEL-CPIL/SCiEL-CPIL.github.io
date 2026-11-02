import time
import math
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Moore bound for even girth g=2R
    R = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(R))

    # Precompute non-backtracking walk coefficient pairs (c1, c2)
    # for a circulant C_n(a,b) with steps {a, -a, b, -b}
    valid_coeffs = set()
    if g > 3:
        states = set()
        # Length 1
        states.add((1, 1, 0))
        states.add((-1, -1, 0))
        states.add((2, 0, 1))
        states.add((-2, 0, -1))

        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in [1, -1, 2, -2]:
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else 1 if step == -2 else 0)
                    # Fix: step -2 means subtract b, so nc2 += -1
                    if step == -2:
                        nc2 = c2 - 1
                    else:
                        nc2 = c2 + (1 if step == 2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    valid_coeffs.add((c1, c2))

    # Sort valid_coeffs by "magnitude" to check most likely collisions first
    # This helps break early when invalid
    sorted_coeffs = sorted(valid_coeffs, key=lambda p: abs(p[0]) + abs(p[1]))

    best_n = None
    best_edges = None

    # Search range for circulant graphs
    # Start from Moore bound, expand outward
    n_max_search = moore_bound + 400

    for n in range(moore_bound, n_max_search + 1):
        if time.time() > deadline - 5:
            break

        found = False

        # For each n, try pairs (a, b)
        # To be efficient, iterate a from 1 to n//2 - 1
        # and b from a+1 to n//2 - 1
        # But this is O(n^2) which is too slow for n=243+.
        # Use random sampling + structured search.

        # First, try a few structured candidates
        # a=1, b=2; a=1, b=3; etc.
        structured_pairs = []
        for a in range(1, min(10, n // 2)):
            for b in range(a + 1, min(a + 10, n // 2)):
                if a + b != n and a != b:
                    structured_pairs.append((a, b))

        # Add random pairs
        num_random = 200
        random_pairs = set(structured_pairs)
        for _ in range(num_random):
            a = random.randint(1, n // 2 - 1)
            b = random.randint(1, n // 2 - 1)
            if a != b and a + b != n:
                random_pairs.add((a, b))

        # Combine and shuffle for variety
        all_pairs = list(random_pairs)
        random.shuffle(all_pairs)

        for a, b in all_pairs:
            if time.time() > deadline - 2:
                break

            # Check connectivity: gcd(n, a, b) == 1
            if math.gcd(math.gcd(n, a), b) != 1:
                continue

            # Check girth: no non-backtracking walk of length 3..g-1 sums to 0 mod n
            valid = True
            for c1, c2 in sorted_coeffs:
                if (c1 * a + c2 * b) % n == 0:
                    valid = False
                    break

            if valid:
                # Construct edges
                edges = set()
                for i in range(n):
                    j1 = (i + a) % n
                    if i < j1:
                        edges.add((i, j1))
                    else:
                        edges.add((j1, i))

                    j2 = (i + b) % n
                    if i < j2:
                        edges.add((i, j2))
                    else:
                        edges.add((j2, i))

                best_n = n
                best_edges = list(edges)
                found = True
                break

        if found:
            return best_n, best_edges

    # Fallback: LUW Construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    queue = [0]
    head = 0

    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))

    return len(order), edges
