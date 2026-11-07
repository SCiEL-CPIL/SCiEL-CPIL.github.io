"""Fast coloring for degree-4 graph with bicolored-forest constraint."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _check_bicolored_forest(coloring, adj):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = P['n']
    colors_used = sorted(set(coloring))
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            parent = list(range(n))
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        return False
                    parent[ru] = rv
    return True


def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)
    
    # Fallback: identity coloring (always valid)
    best = list(range(n))
    best_cnt = n
    
    # Try structured patterns for k=3,4,5,6
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        found = False
        # Linear patterns: color[i] = (a*i + b) % k
        for a in range(1, k + 1):
            for b in range(k):
                if time.time() > deadline:
                    break
                coloring = [(a * i + b) % k for i in range(n)]
                # Check proper coloring
                ok = all(coloring[u] != coloring[v] for u, v in edges)
                if not ok:
                    continue
                if _check_bicolored_forest(coloring, adj):
                    cnt = len(set(coloring))
                    if cnt < best_cnt:
                        best = coloring
                        best_cnt = cnt
                    found = True
                    break
            if found:
                break
        if found and best_cnt <= k:
            break
    
    # If no pattern worked, try a simple greedy + local search
    if best_cnt == n and time.time() < deadline - 5.0:
        # Try 4-coloring with a different pattern
        for k in [4, 5, 6]:
            if time.time() > deadline:
                break
            # Try pattern based on vertex index modulo k with offset
            for offset in range(k):
                if time.time() > deadline:
                    break
                coloring = [(i + offset) % k for i in range(n)]
                ok = all(coloring[u] != coloring[v] for u, v in edges)
                if not ok:
                    continue
                if _check_bicolored_forest(coloring, adj):
                    cnt = len(set(coloring))
                    if cnt < best_cnt:
                        best = coloring
                        best_cnt = cnt
                    break
    
    return [int(c) for c in best]

# EVOLVE-BLOCK-END
