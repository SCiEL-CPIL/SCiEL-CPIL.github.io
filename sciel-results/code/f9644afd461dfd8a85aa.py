"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def solve_csp(n, adj, lists, deadline):
        """
        Solve the list-coloring CSP using backtracking with MRV and forward checking.
        Returns a valid coloring (list of ints) or None if failed before deadline.
        """
        # Domain for each vertex is its list
        domains = [set(lst) for lst in lists]
        assignment = [-1] * n

        # Precompute neighbor lists for faster access
        neigh = [sorted(adj[i]) for i in range(n)]

        def get_mrv_vertex(unassigned):
            """Select unassigned vertex with minimum remaining values."""
            best = -1
            best_size = float('inf')
            for v in unassigned:
                sz = len(domains[v])
                if sz < best_size:
                    best_size = sz
                    best = v
                    if sz <= 1:
                        break
            return best, best_size

        def is_consistent(v, c):
            """Check if assigning color c to v is consistent with neighbors."""
            for u in neigh[v]:
                if assignment[u] != -1:
                    if assignment[u] == c:
                        return False
            return True

        def backtrack(unassigned):
            """Recursive backtracking with MRV and forward checking."""
            if time.time() > deadline:
                return False

            if not unassigned:
                return True

            v, sz = get_mrv_vertex(unassigned)
            if sz == 0:
                return False

            unassigned.remove(v)

            # Try colors in order of least constraining value (LCV)
            colors = sorted(domains[v], key=lambda c: sum(
                1 for u in neigh[v] if u in unassigned and c in domains[u]
            ))

            for c in colors:
                if not is_consistent(v, c):
                    continue

                # Assign
                assignment[v] = c

                # Forward checking: check if any neighbor's domain becomes empty
                saved_domains = {}
                ok = True
                for u in neigh[v]:
                    if u in unassigned:
                        saved_domains[u] = domains[u].copy()
                        if c in domains[u]:
                            domains[u].discard(c)
                            if not domains[u]:
                                ok = False
                                break

                if ok:
                    if backtrack(unassigned):
                        return True

                # Undo
                for u, dom in saved_domains.items():
                    domains[u] = dom

            # Undo assignment
            assignment[v] = -1
            unassigned.add(v)
            return False

        unassigned = set(range(n))
        if backtrack(unassigned):
            return assignment
        return None

    def greedy(n, edges, lists):
        """Fallback greedy coloring."""
        adj = build_adj(n, edges)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    def verify(n, edges, lists, colour):
        """Verify a coloring is valid."""
        for v in range(n):
            if colour[v] not in lists[v]:
                return False
        for u, v in edges:
            if colour[u] == colour[v]:
                return False
        return True

    out = []
    deadline = time.time() + 140  # Safety margin before 150s kill

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Not enough time, use greedy for remaining
            n, edges, lists = instance(splits, seed)
            out.append(greedy(n, edges, lists))
            continue

        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        # Try CSP solver first
        result = solve_csp(n, adj, lists, deadline)

        if result is None:
            # Fall back to greedy
            result = greedy(n, edges, lists)

        # Verify
        if not verify(n, edges, lists, result):
            # Try a few more greedy attempts with different orderings
            found = False
            for attempt in range(10):
                if time.time() > deadline:
                    break
                rng = Random(seed + attempt * 1000)
                order = list(range(n))
                rng.shuffle(order)
                # Sort by degree descending as tiebreaker
                adj = build_adj(n, edges)
                order.sort(key=lambda u: -len(adj[u]))
                colour = [None] * n
                for v in order:
                    taken = {colour[u] for u in adj[v] if colour[u] is not None}
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        colour[v] = rng.choice(free)
                    else:
                        colour[v] = lists[v][0]
                if verify(n, edges, lists, colour):
                    result = colour
                    found = True
                    break

        # Ensure types are correct: must be Python int
        result = [int(c) for c in result]
        out.append(result)

    return out
# EVOLVE-BLOCK-END
