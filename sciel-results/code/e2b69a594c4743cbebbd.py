"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_csp(n, edges, lists):
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use bitmask-based domains for speed
        # Palette is 0..4, so 5 bits needed
        FULL_MASK = (1 << 5) - 1  # 0b11111

        # Convert lists to bitmasks
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask

        # Bit count lookup table (0..31)
        BIT_COUNT = [0] * 32
        for i in range(32):
            BIT_COUNT[i] = i.bit_count()

        # AC-3 with bitmask domains
        def ac3():
            queue = [(i, j) for i in range(n) for j in adj[i]]
            while queue:
                i, j = queue.pop()
                # Revise: remove values from domains[i] that are not in domains[j]
                new_di = domains[i] & domains[j]
                if new_di != domains[i]:
                    domains[i] = new_di
                    if new_di == 0:
                        return False
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
            return True

        # Singleton propagation: if a domain has one value, remove it from all neighbors
        def propagate_singletons():
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if domains[i] and BIT_COUNT[domains[i]] == 1:
                        # i has a single value
                        val_mask = domains[i]
                        for j in adj[i]:
                            if domains[j] & val_mask:
                                domains[j] &= ~val_mask
                                if domains[j] == 0:
                                    return False
                                changed = True
            return True

        # Backtracking with forward checking and singleton propagation
        def backtrack(colour):
            # Find unassigned vertex with minimum remaining values (MRV)
            best_v = -1
            best_count = 999
            for v in range(n):
                if colour[v] is None and domains[v]:
                    cnt = BIT_COUNT[domains[v]]
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                        if cnt == 1:
                            break
                    elif cnt == 1:
                        break

            if best_v == -1:
                # All assigned
                return True

            v = best_v
            # Try each colour in domain
            d = domains[v]
            while d:
                c = (d & -d).bit_length() - 1  # lowest set bit
                d &= d - 1  # clear lowest set bit

                colour[v] = c
                val_mask = 1 << c

                # Forward checking + save
                saved = {}
                valid = True
                for u in adj[v]:
                    if colour[u] is None:
                        saved[u] = domains[u]
                        domains[u] &= ~val_mask
                        if domains[u] == 0:
                            valid = False
                            break

                if valid:
                    # Try singleton propagation
                    if propagate_singletons():
                        if backtrack(colour):
                            return True
                    # Restore after propagation
                    # Re-run forward checking to restore
                    for u in adj[v]:
                        if colour[u] is None:
                            domains[u] = saved[u]

                # Restore
                for u, sv in saved.items():
                    domains[u] = sv
                colour[v] = None

            return False

        # Apply AC-3 first
        if not ac3():
            return None

        # Apply singleton propagation
        if not propagate_singletons():
            return None

        # Check if any domain is empty
        for d in domains:
            if d == 0:
                return None

        # Backtracking search
        colour = [None] * n
        if backtrack(colour):
            return colour
        return None

    def greedy_fallback(n, edges, lists):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            free = None
            for c in lists[v]:
                if c not in taken:
                    free = c
                    break
            colour[v] = free if free is not None else lists[v][0]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we're out of time, return what we have padded with greedy
            break
        n, edges, lists = instance(splits, seed)
        result = solve_csp(n, edges, lists)
        if result is None:
            result = greedy_fallback(n, edges, lists)
        # Ensure all values are Python ints
        result = [int(c) for c in result]
        out.append(result)

    # If we ran out of time, fill remaining with greedy
    while len(out) < 4:
        idx = len(out)
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[idx]
        n, edges, lists = instance(splits, seed)
        result = greedy_fallback(n, edges, lists)
        result = [int(c) for c in result]
        out.append(result)

    return out
# EVOLVE-BLOCK-END
