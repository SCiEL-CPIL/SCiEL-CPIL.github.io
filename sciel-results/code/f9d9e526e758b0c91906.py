"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys

    # Deadline guard: kill after 140s to be safe (evaluator kills at 150s)
    start_time = time.time()
    DEADLINE = 140.0

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by degree descending (most constrained first)
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        pos = {v: i for i, v in enumerate(order)}
        
        # Precompute neighbours in the order (later neighbours only)
        later = [[] for _ in range(n)]
        for v in range(n):
            pv = pos[v]
            for u in adj[v]:
                if pos[u] > pv:
                    later[v].append(u)
        
        colour = [None] * n
        domains = [set(lst) for lst in lists]
        
        # Recursion depth limit
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            if idx == n:
                return True
            if time_remaining() < 0.1:
                return False
                
            v = order[idx]
            used = set()
            for u in adj[v]:
                cu = colour[u]
                if cu is not None:
                    used.add(cu)
            
            # Try colours in domain minus used
            # Sort by "least constraining" heuristic? Or just sorted?
            # Sorted is deterministic and fast.
            for c in sorted(domains[v] - used):
                colour[v] = c
                ok = True
                for u in later[v]:
                    if u in used:
                        continue
                    used_u = set()
                    for w in adj[u]:
                        cw = colour[w]
                        if cw is not None:
                            used_u.add(cw)
                    if not (domains[u] - used_u):
                        ok = False
                        break
                if ok:
                    if backtrack(idx + 1):
                        return True
                colour[v] = None
            return False

        if not backtrack(0):
            # Fallback: should not happen per Alon-Tarsi, but return something valid-length
            # If timeout occurred, this might be called with some vertices coloured.
            # We need to fill in the rest. A simple greedy fill might work or just pick first in list.
            # Since we want valid ints, picking first in list is safest for type, though likely invalid coloring.
            # But the task says "An instance whose colouring breaks a list or an edge scores 0 for that instance".
            # So we just need to return a list of ints.
            for v in range(n):
                if colour[v] is None:
                    colour[v] = lists[v][0]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time_remaining() < 1.0:
            # Not enough time to even generate and solve the next instance reliably
            # But we must return 4 lists. If we haven't started, we have a problem.
            # However, generation is fast (<1ms). Solving is the bottleneck.
            # If we run out of time mid-solve, backtrack returns False, and we fill with defaults.
            pass
            
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
        
    # Ensure types are Python int and lists
    final_out = []
    for lst in out:
        if isinstance(lst, (list, tuple)):
            final_out.append([int(x) for x in lst])
        else:
            final_out.append(lst)
            
    return final_out
# EVOLVE-BLOCK-END
