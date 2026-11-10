"""Model-and-optimize: parameterise the colouring (one int per vertex) and fit
each instance directly against the objective (list-respecting, edge-different
colouring) using a backtracking constraint solver with forward checking."""
# EVOLVE-BLOCK-START
import time
import heapq
import sys
from random import Random

sys.setrecursionlimit(10000)

def construct():
    t0 = time.time()
    LIMIT = 120.0          # hard wall-clock cap for the evaluator
    MARGIN = 10.0          # safety margin before the cap (return early)

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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

    def solve_coloring(n, adj, lists):
        # Initialize available colors for each vertex
        avail = [set(l) for l in lists]
        color = [-1] * n
        colored = [False] * n
        num_colored = 0
        
        # AC-3 arc consistency preprocessing
        import collections
        queue = collections.deque()
        for u in range(n):
            for nb in adj[u]:
                queue.append((u, nb))
        
        while queue:
            u, v = queue.popleft()
            # Check if there exists a color in avail[u] that is not in avail[v]
            # (i.e., v doesn't block all of u's colors)
            # Actually for list coloring with != constraint: 
            # For each color c in avail[u], we need some d in avail[v] with d != c
            # This fails only if avail[v] = {c} (single element equal to c)
            # More precisely: c is supported in v if there exists d in avail[v] with d != c
            # c is NOT supported if avail[v] is a subset of {c}, i.e., avail[v] == {c}
            removed = False
            for c in list(avail[u]):
                # c is not supported by v if all colors in avail[v] equal c
                # i.e., avail[v] is a subset of {c}
                if avail[v] and avail[v].issubset({c}):
                    avail[u].discard(c)
                    removed = True
            if removed and not avail[u]:
                return None
            if removed:
                for nb2 in adj[u]:
                    if nb2 != v:
                        queue.append((nb2, u))
        
        # Check for empty domains after AC-3
        for i in range(n):
            if not avail[i]:
                return None
        
        def backtrack():
            nonlocal num_colored
            if num_colored == n:
                return True
            
            # MRV: find uncolored vertex with fewest available colors
            v = -1
            best = n + 1
            for i in range(n):
                if not colored[i]:
                    cnt = len(avail[i])
                    if cnt == 0:
                        return False
                    if cnt < best:
                        best = cnt
                        v = i
                        if cnt == 1:
                            break
            
            if v == -1:
                return num_colored == n

            # LCV: order colors by impact on neighbors
            colors = list(avail[v])
            if len(colors) > 1:
                def color_impact(c):
                    impact = 0
                    for nb in adj[v]:
                        if not colored[nb] and c in avail[nb]:
                            impact += 1
                    return impact
                colors.sort(key=color_impact)
            
            for c in colors:
                # Try color c
                color[v] = c
                colored[v] = True
                num_colored += 1
                
                # Check neighbors and propagate
                valid = True
                affected = []
                for nb in adj[v]:
                    if colored[nb]:
                        if color[nb] == c:
                            valid = False
                            break
                    else:
                        if c in avail[nb]:
                            avail[nb].remove(c)
                            affected.append(nb)
                            if not avail[nb]:
                                valid = False
                                break
                        
                if valid:
                    if backtrack():
                        return True
                
                # Undo changes
                num_colored -= 1
                colored[v] = False
                color[v] = -1
                for nb in affected:
                    avail[nb].add(c)
            
            return False

        if backtrack():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            break
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        col = solve_coloring(n, adj, lists)
        
        if col is None:
            # Fallback: This should not happen per task description, but just in case
            # Use a simple random assignment that respects lists (might break edges)
            # Or return a dummy valid structure to avoid crash, though it will score 0 for this instance.
            # Better: try a greedy once more with different random seed?
            # Given the theorem, it should always be solvable.
            # Let's just return an empty list or dummy to prevent crash, but we expect non-None.
            col = [lists[i][0] for i in range(n)]
            
        # Ensure types are Python int
        out.append([int(c) for c in col])

    return out
# EVOLVE-BLOCK-END
