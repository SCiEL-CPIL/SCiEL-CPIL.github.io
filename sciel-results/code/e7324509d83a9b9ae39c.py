import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        # Returns list of maximal cliques as frozensets
        cliques = []
        # Precompute neighbor sets
        neigh = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    neigh[u].add(v)
                    neigh[v].add(u)

        def expand(R, P, X):
            if not P and not X:
                cliques.append(frozenset(R))
                return
            # Pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            union = P | X
            if not union:
                return
            best_u = None
            best_count = -1
            for u in union:
                cnt = len(P & neigh[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            # Iterate over P \ N(best_u)
            for v in list(P - neigh[best_u]):
                expand(R | {v}, P & neigh[v], X & neigh[v])
                P.remove(v)
                X.add(v)

        expand(set(), set(range(n)), set())
        return cliques

    def is_valid_sol(sol_set, cliques):
        sol_set = set(sol_set)
        for c in cliques:
            if not sol_set & c:
                return False
        return True

    def greedy_hit_set(cliques, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        uncovered = [set(c) for c in cliques]
        selected = []
        while True:
            if all(len(c) == 0 for c in uncovered):
                break
            # Count hits for each vertex
            hits = [0] * n
            for c in uncovered:
                for v in c:
                    hits[v] += 1
            if max(hits) == 0:
                break
            v = max(range(n), key=lambda i: hits[i])
            selected.append(v)
            for i, c in enumerate(uncovered):
                if v in c:
                    c.discard(v)
        return selected

    def branch_and_bound(cliques, n, time_limit):
        """
        Custom branch-and-bound for hitting set (set cover).
        Uses bitmasks for speed.
        """
        m = len(cliques)
        # Represent each clique as a bitmask of vertices
        clique_masks = [0] * m
        for i, c in enumerate(cliques):
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks[i] = mask

        # For each vertex, precompute which cliques it hits
        vertex_cliques = [[] for _ in range(n)]
        for i in range(m):
            mask = clique_masks[i]
            for v in range(n):
                if mask & (1 << v):
                    vertex_cliques[v].append(i)

        best_sol = None
        best_size = n + 1

        # Greedy initial solution
        greedy = greedy_hit_set(cliques, n)
        best_size = len(greedy)
        best_sol = greedy

        # Sort vertices by degree (number of cliques hit) descending for branching
        vertex_order = sorted(range(n), key=lambda v: -len(vertex_cliques[v]))

        # State: (selected_mask, uncovered_cliques_mask)
        # uncovered_cliques_mask: bitmask of cliques still not hit
        initial_uncovered = (1 << m) - 1

        def bb(selected_mask, uncovered_mask, depth):
            nonlocal best_sol, best_size

            if time.time() > start_time + time_limit:
                return

            if uncovered_mask == 0:
                # All cliques covered
                sol_size = bin(selected_mask).count('1')
                if sol_size < best_size:
                    best_size = sol_size
                    best_sol = [v for v in range(n) if selected_mask & (1 << v)]
                return

            # Lower bound: at least 1 more vertex needed if uncovered_mask != 0
            # Better LB: count uncovered cliques, each vertex can cover at most max_cliques_per_vertex
            # Simple LB: 1 (if uncovered)
            # Even better: greedy lower bound on remaining
            if bin(selected_mask).count('1') + 1 >= best_size:
                return

            # Try to find a vertex that covers the most uncovered cliques
            # For branching, pick a vertex from the first uncovered clique
            # Find first uncovered clique
            # Get its index
            first_clique_idx = None
            for i in range(m):
                if uncovered_mask & (1 << i):
                    first_clique_idx = i
                    break

            if first_clique_idx is None:
                return

            # Branch on vertices in this clique
            clique_mask = clique_masks[first_clique_idx]
            candidates = []
            for v in range(n):
                if clique_mask & (1 << v):
                    candidates.append(v)

            # Sort candidates by how many additional uncovered cliques they cover
            def score(v):
                count = 0
                for ci in vertex_cliques[v]:
                    if uncovered_mask & (1 << ci):
                        count += 1
                return -count  # Minimize, so more coverage = lower score

            candidates.sort(key=score)

            for v in candidates:
                if time.time() > start_time + time_limit:
                    return
                new_selected = selected_mask | (1 << v)
                # Update uncovered: remove all cliques hit by v
                new_uncovered = uncovered_mask
                for ci in vertex_cliques[v]:
                    new_uncovered &= ~(1 << ci)

                # Prune: if selected size + 1 >= best_size, skip
                if bin(new_selected).count('1') >= best_size:
                    continue

                bb(new_selected, new_uncovered, depth + 1)

        bb(0, initial_uncovered, 0)

        if best_sol is not None:
            return best_sol
        return greedy

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]

        if not relevant_cliques:
            return []

        # Allocate time based on remaining budget
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 10:
            return greedy_hit_set(relevant_cliques, n)

        # Give each instance a fair share
        instance_time = min(35, max(5, (time_left - 5) / 3))

        sol = branch_and_bound(relevant_cliques, n, instance_time)

        # Verify
        if not is_valid_sol(sol, relevant_cliques):
            sol = greedy_hit_set(relevant_cliques, n)

        return sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))

        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results
