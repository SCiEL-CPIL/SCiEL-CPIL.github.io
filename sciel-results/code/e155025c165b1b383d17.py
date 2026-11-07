"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

sys.setrecursionlimit(100000)

def construct():
    from random import Random

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

    def arc_consistency(n, adj, domains, deadline):
        """AC-3: enforce arc consistency. Returns False if any domain becomes empty."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    queue.append((u, v))
                    queue.append((v, u))
        while queue:
            if time.time() > deadline:
                return True  # stop pruning, keep what we have
            u, v = queue.popleft()
            removed = False
            to_remove = []
            for cu in domains[u]:
                if not any(cu != cv for cv in domains[v]):
                    to_remove.append(cu)
            for cu in to_remove:
                domains[u].discard(cu)
                removed = True
            if removed:
                if not domains[u]:
                    return False
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))
        return True

    def solve_csp(n, adj, lists, deadline):
        """MRV + forward checking backtracking with AC preprocessing."""
        domains = [set(l) for l in lists]
        
        # AC-3 preprocessing
        if not arc_consistency(n, adj, domains, deadline):
            return None
        
        # Check if any domain is empty after AC
        for v in range(n):
            if not domains[v]:
                return None
        
        uncoloured = set(range(n))
        colour = [None] * n
        degree = [len(adj[v]) for v in range(n)]

        def mrv_vertex():
            best = None
            best_size = 100
            best_deg = -1
            for v in uncoloured:
                s = len(domains[v])
                if s < best_size or (s == best_size and degree[v] > best_deg):
                    best_size = s
                    best_deg = degree[v]
                    best = v
                    if s <= 1:
                        break
            return best

        def propagate(v, c):
            removed = []
            for u in adj[v]:
                if u in uncoloured and c in domains[u]:
                    domains[u].discard(c)
                    removed.append((u, c))
                    if not domains[u]:
                        return None
            return removed

        def backtrack():
            if time.time() > deadline:
                return None
            if not uncoloured:
                return True
            v = mrv_vertex()
            if v is None:
                return True
            # Least constraining value heuristic
            for c in sorted(domains[v], key=lambda c: sum(1 for u in adj[v] if u in uncoloured and c in domains[u])):
                uncoloured.discard(v)
                colour[v] = c
                removed = propagate(v, c)
                if removed is None:
                    uncoloured.add(v)
                    colour[v] = None
                    continue
                result = backtrack()
                if result is not None:
                    return result
                for u, rc in removed:
                    domains[u].add(rc)
                uncoloured.add(v)
                colour[v] = None
            return None

        result = backtrack()
        return result

    def greedy_with_retry(n, adj, lists, num_retries=50):
        """Greedy coloring with multiple random orderings as fallback."""
        from random import Random
        rng = Random(42)
        best_colour = None
        best_violations = n * n  # large number
        
        for attempt in range(num_retries):
            colour = [None] * n
            # Random perturbation of degree ordering
            order = sorted(range(n), key=lambda u: (-len(adj[u]), rng.random()))
            valid = True
            for v in order:
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                if free:
                    colour[v] = free[0]
                else:
                    # Conflict: pick least-used color from list
                    colour[v] = min(lists[v], key=lambda c: sum(1 for u in adj[v] if colour[u] == c))
            
            # Count violations
            violations = 0
            for v in range(n):
                if colour[v] not in lists[v]:
                    violations += 1
                for u in adj[v]:
                    if u > v and colour[u] == colour[v]:
                        violations += 1
            
            if violations == 0:
                return colour
            if violations < best_violations:
                best_violations = violations
                best_colour = list(colour)
        
        return best_colour

    start = time.time()
    deadline = start + 140.0  # 10s safety margin before 150s kill

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        remaining = deadline - time.time()
        if remaining < 3.0:
            # Not enough time for full solve; use fast greedy
            colouring = greedy_with_retry(n, adj, lists, num_retries=10)
            out.append([int(c) for c in colouring])
            continue

        # Allocate time: more for harder instances
        time_limit = min(remaining - 2.0, 30.0)
        if splits >= 60:
            time_limit = min(time_limit, remaining - 2.0)
        
        # Try CSP solver
        result = solve_csp(n, adj, lists, start + time_limit)
        if result is not None:
            # Validate
            valid = True
            for v in range(n):
                if result[v] not in lists[v]:
                    valid = False
                    break
                for u in adj[v]:
                    if u > v and result[u] == result[v]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                out.append([int(c) for c in result])
                continue

        # Fallback: greedy with retry
        remaining_for_greedy = deadline - time.time()
        num_retries = max(10, int(remaining_for_greedy * 5))
        colouring = greedy_with_retry(n, adj, lists, num_retries=num_retries)
        out.append([int(c) for c in colouring])

    # Ensure all are Python ints in lists
    return [[int(c) for c in lst] for lst in out]

# EVOLVE-BLOCK-END
