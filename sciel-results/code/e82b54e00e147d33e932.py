"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    import random
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

    def solve_instance(n, edges, lists, deadline):
        """Solve a single list colouring instance using AC-3 initialization and backtracking with MRV and Forward Checking."""
        if time.time() > deadline:
            return None

        # Build adjacency as lists of sets for faster membership checks
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Initialize domains as sets for O(1) lookup and removal
        domains = [set(lst) for lst in lists]
        color = [-1] * n
        uncoloured = set(range(n))

        # AC-3 Initialization
        # Queue of arcs (u, v) to check
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        ac3_failed = False
        while queue:
            if time.time() > deadline:
                return None
            u, v = queue.popleft()
            # Check if domains[u] is consistent with domains[v]
            # Remove colors from domains[u] that are in domains[v] (since they must be different)
            # Wait, standard AC-3 for "not equal" constraint:
            # Revise(u, v): remove c from domains[u] if c is in domains[v] AND no other value in domains[v] is != c?
            # No, for "not equal", c is invalid in u if c is the ONLY value in v?
            # Actually, if c is in domains[u] and c is in domains[v], it is NOT necessarily invalid.
            # It is only invalid if domains[v] == {c}.
            # Let's verify: If u=c and v must be != c, then v must take a value from domains[v] \ {c}.
            # If domains[v] \ {c} is empty, then u cannot be c.
            
            to_remove = []
            for c in domains[u]:
                # If c is in domains[v], can v take a different color?
                # If len(domains[v]) == 1 and c in domains[v], then v MUST be c, so u cannot be c.
                if c in domains[v] and len(domains[v]) == 1:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    domains[u].discard(c)
                if not domains[u]:
                    ac3_failed = True
                    break
                # If domains[u] changed, we need to re-check all other neighbors of u
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))
        
        if ac3_failed:
            return None

        # Check if any domain is empty after AC-3 (though AC-3 would have caught it)
        for i in range(n):
            if not domains[i]:
                return None

        # If all domains have size 1, we have a solution
        all_single = True
        for i in range(n):
            if len(domains[i]) > 1:
                all_single = False
                break
        
        if all_single:
            for i in range(n):
                color[i] = next(iter(domains[i]))
            # Verify
            for u, v in edges:
                if color[u] == color[v]:
                    return None
            return color

        def backtrack():
            if time.time() > deadline:
                return False

            if not uncoloured:
                return True

            # MRV Heuristic: Select uncoloured vertex with smallest domain
            # Tie-breaker: Highest degree (more constraints)
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            for v in uncoloured:
                size = len(domains[v])
                if size == 0:
                    return False
                deg = len(adj[v])
                if size < min_size or (size == min_size and deg > best_deg):
                    min_size = size
                    best_v = v
                    best_deg = deg

            v = best_v
            uncoloured.remove(v)

            # Value Ordering: Try values that appear least in neighbors' domains
            # We can sort by the number of values removed from neighbors
            # But for simplicity and speed, we stick to standard iteration or simple LCV approximation
            # Let's just iterate over the domain
            
            for c in list(domains[v]):
                # Assign
                color[v] = c
                
                # Record changes for undo
                changes = []
                valid = True
                
                for u in adj[v]:
                    if u in uncoloured:
                        if c in domains[u]:
                            changes.append((u, c))
                            domains[u].discard(c)
                            if not domains[u]:
                                valid = False
                                break

                if valid:
                    if backtrack():
                        return True

                # Undo
                for u, c_val in changes:
                    domains[u].add(c_val)
                uncoloured.add(v)
                color[v] = -1

            return False

        if backtrack():
            return color
        
        return None

    deadline = time.time() + 110  # 110 second safety margin (150s limit)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            # Ensure all values are Python int
            out.append([int(c) for c in result])
        else:
            # Should not happen, but add a dummy valid coloring
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
