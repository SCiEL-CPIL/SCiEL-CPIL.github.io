"""Colouring four fixed planar bipartite graphs from lists of three colours."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict


def construct():
    import random
    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b))
                 for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]),
                           max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists, deadline):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for a in adj:
            a.sort()

        order = sorted(range(n), key=lambda v: (len(adj[v]), v))
        colour = [-1] * n
        domains = [list(lst) for lst in lists]

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True

            # MRV with limited lookahead
            best_idx = idx
            best_size = len(domains[order[idx]])
            for j in range(idx + 1, min(idx + 6, n)):
                s = len(domains[order[j]])
                if s < best_size:
                    best_idx = j
                    best_size = s
                    if s <= 1:
                        break
            if best_idx != idx:
                order[idx], order[best_idx] = order[best_idx], order[idx]

            v = order[idx]
            dom = domains[v]
            neighbours = adj[v]
            
            # LCV
            lcv_order = []
            for c in dom:
                cnt = 0
                for u in neighbours:
                    if colour[u] == -1 and c in domains[u]:
                        cnt += 1
                lcv_order.append((cnt, c))
            lcv_order.sort()

            for _, c in lcv_order:
                if any(colour[u] == c for u in neighbours):
                    continue

                colour[v] = c
                saved = []
                feasible = True
                for u in neighbours:
                    if colour[u] == -1:
                        du = domains[u]
                        if c in du:
                            new_du = du.copy()
                            new_du.remove(c)
                            saved.append((u, du))
                            if not new_du:
                                feasible = False
                                break
                            domains[u] = new_du
                
                if feasible:
                    if backtrack(idx + 1):
                        return True
                
                for u, old_du in saved:
                    domains[u] = old_du
                colour[v] = -1

            return False

        if backtrack(0):
            return colour

        # Fallback
        order[:] = sorted(range(n), key=lambda v: (-len(adj[v]), v))
        colour[:] = [-1] * n
        domains = [list(lst) for lst in lists]
        if backtrack(0):
            return colour

        return None

    start = time.time()
    deadline = start + 110

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            out.append([int(c) for c in result])
        else:
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = set()
                for u in adj[v]:
                    taken.add(colour[u])
                for c in lists[v]:
                    if c not in taken:
                        colour[v] = c
                        break
            out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
