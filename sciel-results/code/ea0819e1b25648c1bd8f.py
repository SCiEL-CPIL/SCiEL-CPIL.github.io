"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import time as _time

    deadline = _time.monotonic() + 130.0

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

    def solve_csp(n, edges, color_lists):
        """Backtracking CSP solver with forward checking, MRV, and incremental domain restoration."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Domains: each vertex starts with its list of 3 colours
        orig_domains = [list(color_lists[v]) for v in range(n)]
        domains = [set(color_lists[v]) for v in range(n)]
        assignment = [None] * n
        
        # Local references for speed
        _adj = adj
        _assignment = assignment
        _domains = domains
        _deadline = deadline

        def forward_check(v, c):
            """Remove c from domains of unassigned neighbors. Return list of (neighbor, old_domain) for undo, or None if wipeout."""
            changes = []
            for u in _adj[v]:
                if _assignment[u] is None and c in _domains[u]:
                    old = _domains[u]
                    new = old - {c}
                    if not new:
                        # Undo changes already made
                        for (uu, old_dom) in changes:
                            _domains[uu] = old_dom
                        return None
                    changes.append((u, old))
                    _domains[u] = new
            return changes

        def undo_forward_check(changes):
            """Restore domains from changes list."""
            for (u, old_dom) in changes:
                _domains[u] = old_dom

        def get_mrv_vertex():
            """Pick unassigned vertex with minimum remaining values."""
            min_size = float('inf')
            best = -1
            for v in range(n):
                if _assignment[v] is None:
                    sz = len(_domains[v])
                    if sz == 0:
                        return v  # domain wipeout
                    if sz < min_size:
                        min_size = sz
                        best = v
            return best

        def backtrack(depth):
            if _time.monotonic() > _deadline:
                return None  # timeout

            # Find unassigned vertex with MRV
            v = get_mrv_vertex()
            if v == -1:
                # All assigned
                return _assignment[:]

            # Try each colour in domain
            d = _domains[v]
            # Sort colours by how many neighbours they eliminate (fewer eliminations first)
            def count_eliminations(c):
                count = 0
                for u in _adj[v]:
                    if _assignment[u] is None and c in _domains[u]:
                        count += 1
                return count
            sorted_colors = sorted(d, key=count_eliminations)

            for c in sorted_colors:
                _assignment[v] = c
                changes = forward_check(v, c)
                if changes is not None:
                    result = backtrack(depth + 1)
                    if result is not None:
                        return result
                else:
                    # forward_check already undid its own changes on wipeout
                    pass
                undo_forward_check(changes if changes is not None else [])
                _assignment[v] = None

            return None

        # Try with MRV backtracking
        result = backtrack(0)
        if result is not None:
            return result

        # Fallback: simple greedy in case of timeout or failure
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in color_lists[v] if c not in taken]
            colour[v] = free[0] if free else color_lists[v][0]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if _time.monotonic() > deadline:
            # Not enough time; return greedy fallback for remaining
            n, edges, lists = instance(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
            continue
        n, edges, lists = instance(splits, seed)
        out.append(solve_csp(n, edges, lists))
    return out
# EVOLVE-BLOCK-END
