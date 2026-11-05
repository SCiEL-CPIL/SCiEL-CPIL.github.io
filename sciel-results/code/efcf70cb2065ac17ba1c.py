import math
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    N = 511  # vertices 1..511

    # Weights: w[a] = 1/(512-a)
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)

    # Build conflict graph: edge between a,b if gcd(a,b) > 1
    adjacency = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if math.gcd(a, b) > 1:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # Order vertices by weight descending
    order = list(range(N, 0, -1))  # 511, 510, ..., 1

    # Current best
    best_set = set()
    best_score = 0.0

    # Greedy coloring upper bound for maximum weight independent set
    def greedy_coloring_ub(available):
        if not available:
            return 0.0
        # Sort by weight descending
        verts = sorted(available, key=lambda x: -weights[x])
        color_of = {}
        for v in verts:
            used = set()
            for u in adjacency[v]:
                if u in available and u in color_of:
                    used.add(color_of[u])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
        # Sum of max weight per color class
        max_w = {}
        for v, c in color_of.items():
            if c not in max_w or weights[v] > max_w[c]:
                max_w[c] = weights[v]
        return sum(max_w.values())

    def bb(idx, current_set, current_score, available):
        nonlocal best_set, best_score, deadline, start_time
        if time.time() > deadline:
            return
        if idx >= len(order):
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        # Upper bound using greedy coloring
        ub = current_score + greedy_coloring_ub(available)
        if ub <= best_score + 1e-12:
            return
        a = order[idx]
        # Option 1: Include a
        if a in available:
            if all(math.gcd(a, c) == 1 for c in current_set):
                new_avail = available - {a}
                for conf in adjacency[a]:
                    new_avail.discard(conf)
                bb(idx + 1, current_set | {a}, current_score + weights[a], new_avail)
        # Option 2: Exclude a
        new_avail = available - {a}
        bb(idx + 1, current_set, current_score, new_avail)

    # Initial greedy solution
    cur = set()
    sc = 0.0
    for a in order:
        if all(math.gcd(a, c) == 1 for c in cur):
            cur.add(a)
            sc += weights[a]
    best_set = set(cur)
    best_score = sc

    # Run B&B
    initial_available = set(range(1, N + 1))
    bb(0, set(), 0.0, initial_available)

    return list(best_set)
