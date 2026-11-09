import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s hard kill; keep 5s margin

    n = P['n']
    edges = P['edges']

    # Original adjacency (for validation only)
    orig_adj = [set() for _ in range(n)]
    for u, v in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)

    def expand_max_biclique(u, v, adj):
        """Expand seed edge (u,v) to a maximal biclique in graph adj (both orientations)."""
        def expand(L0, R0):
            L = set(L0)
            R = set(R0)
            while True:
                newL = set()
                for x in L:
                    if R.issubset(adj[x]):
                        newL.add(x)
                newR = set()
                for y in R:
                    if newL.issubset(adj[y]):
                        newR.add(y)
                if len(newL) == len(L) and len(newR) == len(R):
                    return newL, newR
                L, R = newL, newR

        # Orientation 1: u in L, v in R
        L1 = set(adj[v])
        R1 = set(adj[u])
        if u in L1 and v in R1:
            L1, R1 = expand(L1, R1)
            s1 = len(L1) * len(R1)
        else:
            s1 = 0
        # Orientation 2: u in R, v in L
        L2 = set(adj[u])
        R2 = set(adj[v])
        if v in L2 and u in R2:
            L2, R2 = expand(L2, R2)
            s2 = len(L2) * len(R2)
        else:
            s2 = 0

        if s1 >= s2:
            return list(L1), list(R1), s1
        else:
            return list(L2), list(R2), s2

    def matching_lower_bound(adj):
        """Max matching in adj as a lower bound on biclique cover number."""
        match = [-1] * n
        def try_kuhn(u, seen):
            for w in adj[u]:
                if seen[w]:
                    continue
                seen[w] = True
                if match[w] == -1 or try_kuhn(match[w], seen):
                    match[w] = u
                    return True
            return False
        cnt = 0
        for u in range(n):
            if match[u] == -1:
                seen = [False] * n
                if try_kuhn(u, seen):
                    cnt += 1
        return cnt

    def exact_bnb(rem_edges, adj, best_limit, time_budget):
        """Branch-and-bound to partition rem_edges into <= best_limit bicliques.
        Returns list of [L,R] or None."""
        if not rem_edges:
            return []
        t0 = time.time()
        node = 0

        # Precompute all maximal bicliques for the residual graph
        mb = []
        for (a, b) in rem_edges:
            L, R, s = expand_max_biclique(a, b, adj)
            if s >= 1:
                mb.append((set(L), set(R)))
        if not mb:
            return None

        # Deduplicate
        uniq = []
        seen = set()
        for L, R in mb:
            key = (frozenset(L), frozenset(R))
            if key not in seen:
                seen.add(key)
                uniq.append((L, R))
        mb = uniq

        best = [None]
        best_count = [best_limit]
        lb0 = matching_lower_bound(adj)
        if lb0 > best_limit:
            return None

        def rec(rem_set, adj_cur, count):
            nonlocal node
            node += 1
            if time.time() - t0 > time_budget:
                return
            if not rem_set:
                if count < best_count[0]:
                    best_count[0] = count
                return
            lb = count + matching_lower_bound(adj_cur)
            if lb >= best_count[0]:
                return
            # pick a residual edge
            e = next(iter(rem_set))
            a, b = e
            for (L, R) in mb:
                # biclique must contain e
                if a not in L or b not in R:
                    continue
                # all cross edges must be in rem_set
                ok = True
                for x in L:
                    for y in R:
                        key = (min(x, y), max(x, y))
                        if key not in rem_set:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                new_rem = set(rem_set)
                new_adj = [set(s) for s in adj_cur]
                for x in L:
                    for y in R:
                        key = (min(x, y), max(x, y))
                        if key in new_rem:
                            new_rem.discard(key)
                            new_adj[x].discard(y)
                            new_adj[y].discard(x)
                rec(new_rem, new_adj, count + 1)
                if best_count[0] <= count + 1:
                    # cannot improve further with this branch depth
                    pass

        rec(set(rem_edges), [set(s) for s in adj], 0)
        return best[0]

    def greedy_solve(seed, use_exact_tail=True):
        random.seed(seed)
        t0 = time.time()

        uncovered = [set() for _ in range(n)]
        for u, v in edges:
            uncovered[u].add(v)
            uncovered[v].add(u)

        # current remaining adjacency
        adj = [set(s) for s in orig_adj]

        bicliques = []

        EXACT_THRESHOLD = 14  # number of residual edges to trigger exact solve

        while any(uncovered[u] for u in range(n)):
            if time.time() > deadline - 2:
                return None

            # Build list of remaining edges (canonical)
            rem_edges = set()
            for u in range(n):
                for v in uncovered[u]:
                    if u < v:
                        rem_edges.add((u, v))
            if not rem_edges:
                break

            # If small, try exact
            if use_exact_tail and len(rem_edges) <= EXACT_THRESHOLD:
                exact_time = min(8.0, max(1.0, deadline - 3 - time.time()))
                if exact_time > 0.5:
                    sol = exact_bnb(list(rem_edges), adj, best_limit=50,
                                    time_budget=exact_time)
                    if sol is not None:
                        bicliques.extend(sol)
                        return bicliques

            # Greedy step: pick a seed edge, expand to maximal biclique
            # Sample several seed edges and pick best expansion
            seed_edges = list(rem_edges)
            random.shuffle(seed_edges)
            sample = seed_edges[:min(40, len(seed_edges))]

            best_L, best_R, best_size = None, None, -1
            for (a, b) in sample:
                L, R, s = expand_max_biclique(a, b, adj)
                if s > best_size:
                    best_size = s
                    best_L, best_R = L, R

            if best_L is None or best_size < 1:
                # fallback single edge
                a, b = next(iter(rem_edges))
                best_L, best_R = [a], [b]

            bicliques.append([list(best_L), list(best_R)])

            # Update uncovered and adj
            for x in best_L:
                for y in best_R:
                    if y in uncovered[x]:
                        uncovered[x].discard(y)
                    if x in uncovered[y]:
                        uncovered[y].discard(x)
                    adj[x].discard(y)
                    adj[y].discard(x)

        return bicliques

    def is_valid(bicliques):
        if not bicliques:
            return False
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            Ls = set(L)
            Rs = set(R)
            if Ls & Rs:
                return False
            for x in L:
                for y in R:
                    if y not in orig_adj[x]:
                        return False
                    key = (min(x, y), max(x, y))
                    if key in covered:
                        return False
                    covered.add(key)
        all_edges = set()
        for u, v in edges:
            all_edges.add((min(u, v), max(u, v)))
        return covered == all_edges

    best_bicliques = None
    best_count = float('inf')

    seed = 42
    while time.time() < deadline - 3:
        result = greedy_solve(seed)
        if result is not None and is_valid(result):
            count = len(result)
            if count < best_count:
                best_count = count
                best_bicliques = result
                # If we hit a very good value, keep searching a bit
                if best_count <= 10:
                    pass
        seed += 1
        if time.time() > deadline - 5:
            break

    if best_bicliques is None:
        # Trivial valid fallback: each edge its own biclique
        trivial = []
        for u, v in edges:
            trivial.append([[int(u)], [int(v)]])
        return trivial

    # Final type conversion
    result = []
    for L, R in best_bicliques:
        result.append([[int(x) for x in L], [int(y) for y in R]])
    return result
