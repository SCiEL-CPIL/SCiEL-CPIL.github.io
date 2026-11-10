# EVOLVE-BLOCK-START
import time
import sys
from random import Random

sys.setrecursionlimit(100000)

def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 15.0

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
        # Bitmask representation for available colors (0-4)
        avail = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            avail[i] = mask

        color = [-1] * n
        colored = [False] * n
        num_colored = 0

        # Precompute degree for tie-breaking
        degree = [len(adj[i]) for i in range(n)]

        def backtrack():
            nonlocal num_colored
            if num_colored == n:
                return True

            # MRV Heuristic: Pick uncolored vertex with fewest available colors
            # Tie-break by degree (higher degree first)
            min_avail_count = 5
            best_v = -1
            best_deg = -1
            for i in range(n):
                if not colored[i]:
                    cnt = bin(avail[i]).count('1')
                    if cnt == 0:
                        return False
                    if cnt < min_avail_count or (cnt == min_avail_count and degree[i] > best_deg):
                        min_avail_count = cnt
                        best_v = i
                        best_deg = degree[i]

            v = best_v
            mask = avail[v]
            while mask:
                c_bit = mask & -mask
                c = c_bit.bit_length() - 1
                mask ^= c_bit

                color[v] = c
                colored[v] = True
                num_colored += 1

                valid = True
                affected = []
                for nb in adj[v]:
                    if colored[nb]:
                        if color[nb] == c:
                            valid = False
                            break
                    else:
                        if avail[nb] & c_bit:
                            avail[nb] ^= c_bit
                            if avail[nb] == 0:
                                valid = False
                                break
                            affected.append(nb)

                if valid:
                    if backtrack():
                        return True

                num_colored -= 1
                colored[v] = False
                color[v] = -1
                for nb in affected:
                    avail[nb] |= c_bit

        for i in range(n):
            if avail[i] == 0:
                return None

        if backtrack():
            return color
        else:
            return None

    def solve_ac3_backtrack(n, adj, lists):
        # AC-3 preprocessing then backtracking with forward checking
        # Domains as bitmasks
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask

        # AC-3: enforce arc consistency
        # Arc (i, j) means: for each color in domain[i], there exists a color in domain[j] != color
        # Since colors are just labels and constraint is !=, arc consistency for binary != constraint:
        # domain[i] can only be pruned if domain[j] has size 1 and that single color is in domain[i]
        # Actually for !=: remove color c from domain[i] if c is the ONLY color in domain[j]
        # So AC-3: for arc (i,j), if |domain[j]| == 1, remove that color from domain[i]
        
        from collections import deque
        queue = deque()
        for i in range(n):
            for j in adj[i]:
                queue.append((i, j))
        
        while queue:
            i, j = queue.popleft()
            # Check if domain[i] can be revised with respect to domain[j]
            # For != constraint: remove c from domain[i] if c is the only color in domain[j]
            dj = domains[j]
            if dj != 0 and (dj & (dj - 1)) == 0:  # dj is a power of 2, i.e., single color
                c_bit = dj
                if domains[i] & c_bit:
                    domains[i] ^= c_bit
                    if domains[i] == 0:
                        return None
                    # Add all arcs (k, i) where k != j
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
        
        # Now do backtracking with AC-3-style forward checking
        color = [-1] * n
        colored = [False] * n
        num_colored = 0
        degree = [len(adj[i]) for i in range(n)]
        
        def backtrack():
            nonlocal num_colored
            if num_colored == n:
                return True
            
            min_avail_count = 5
            best_v = -1
            best_deg = -1
            for i in range(n):
                if not colored[i]:
                    cnt = bin(domains[i]).count('1')
                    if cnt == 0:
                        return False
                    if cnt < min_avail_count or (cnt == min_avail_count and degree[i] > best_deg):
                        min_avail_count = cnt
                        best_v = i
                        best_deg = degree[i]
            
            v = best_v
            mask = domains[v]
            while mask:
                c_bit = mask & -mask
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                color[v] = c
                colored[v] = True
                num_colored += 1
                
                valid = True
                affected = []
                for nb in adj[v]:
                    if colored[nb]:
                        if color[nb] == c:
                            valid = False
                            break
                    else:
                        if domains[nb] & c_bit:
                            domains[nb] ^= c_bit
                            if domains[nb] == 0:
                                valid = False
                                break
                            affected.append(nb)
                
                if valid:
                    if backtrack():
                        return True
                
                num_colored -= 1
                colored[v] = False
                color[v] = -1
                for nb in affected:
                    domains[nb] |= c_bit
        
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
        
        # Try AC-3 + backtracking first (more robust), fall back to plain backtracking
        col = solve_ac3_backtrack(n, adj, lists)
        
        if col is None:
            col = solve_coloring(n, adj, lists)
        
        if col is None:
            col = [lists[i][0] for i in range(n)]
        
        out.append([int(c) for c in col])

    return out
# EVOLVE-BLOCK-END
