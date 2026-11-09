def construct():
    import time
    import math

    deadline = time.time() + 55
    LIMIT = 512
    N = LIMIT

    # Build adjacency as list of sets for fast intersection
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i // math.gcd(i, j)) * j <= LIMIT:
                adj[i].add(j)
                adj[j].add(i)

    # Precompute sorted neighbor lists for faster iteration
    adj_list = [sorted(adj[i]) for i in range(N + 1)]

    best_clique = []
    best_size = 0

    def color_sort(P):
        if not P:
            return []

        P_list = list(P)
        # Sort by degree in P descending
        P_list.sort(key=lambda v: -len(adj[v] & P))

        color_of = {}
        for v in P_list:
            used_colors = set()
            for u in adj_list[v]:
                if u in color_of:
                    used_colors.add(color_of[u])
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c

        items = [(v, color_of[v]) for v in P_list]
        items.sort(key=lambda x: (-x[1], -len(adj[x[0]] & P)))
        return items

    def expand(C, P):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return

        if not P:
            if len(C) > best_size:
                best_clique = list(C)
                best_size = len(C)
            return

        if len(C) + len(P) <= best_size:
            return

        P_sorted = color_sort(P)

        for i in range(len(P_sorted) - 1, -1, -1):
            if time.time() > deadline:
                return
            v, c = P_sorted[i]

            if len(C) + c <= best_size:
                return

            C.add(v)
            new_P = P & adj[v]
            expand(C, new_P)
            C.discard(v)
            P.discard(v)

    def greedy_clique(start_order=None):
        C = set()
        P = set(range(1, N + 1))
        if start_order is not None:
            P = set(start_order)
        while P:
            v = max(P, key=lambda x: len(adj[x] & P))
            C.add(v)
            P = P & adj[v]
        return C

    # Multiple greedy restarts with different orderings
    init_clique = greedy_clique()
    best_clique = list(init_clique)
    best_size = len(best_clique)

    # Try a few more greedy variants with shuffled-ish orderings
    for seed in range(5):
        import random
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        # Use the shuffled order to break ties in greedy
        C = set()
        P = set(range(1, N + 1))
        while P:
            # Pick vertex with max degree in P, tie-break by position in shuffled order
            v = max(P, key=lambda x: (len(adj[x] & P), -order.index(x)))
            C.add(v)
            P = P & adj[v]
        if len(C) > best_size:
            best_clique = list(C)
            best_size = len(C)

    # Run Tomita
    expand(set(), set(range(1, N + 1)))

    return best_clique
