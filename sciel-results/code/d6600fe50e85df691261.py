"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices by degree (descending) to constrain early
    vertices = list(range(n))
    vertices.sort(key=lambda x: -len(adj[x]))
    
    def has_bicolored_cycle(coloring):
        """Check if the current coloring has any bicolored cycle."""
        from collections import defaultdict
        color_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu == -1 or cv == -1:
                continue
            if cu == cv:
                # Invalid state, but prevented by caller
                continue
            pair = (min(cu, cv), max(cu, cv))
            color_edges[pair].append((u, v))
            
        for pair, elist in color_edges.items():
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u, v in elist:
                if not union(u, v):
                    return True
        
        return False

    def is_adjacent_conflict(v, c, coloring):
        """Check if color c is used by any assigned neighbor of v."""
        for u in adj[v]:
            if coloring[u] == c:
                return True
        return False

    def backtrack(idx, coloring):
        """Recursive backtracking solver."""
        if idx == n:
            return True
        
        v = vertices[idx]
        
        for c in range(k):
            if is_adjacent_conflict(v, c, coloring):
                continue
            
            coloring[v] = c
            
            if not has_bicolored_cycle(coloring):
                if backtrack(idx + 1, coloring):
                    return True
            
            coloring[v] = -1
            
        return False

    for k in range(2, 10):
        coloring = [-1] * n
        if backtrack(0, coloring):
            # Verify solution
            valid = True
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    valid = False
                    break
            if valid and not has_bicolored_cycle(coloring):
                used = sorted(set(coloring))
                remap = {old: new for new, old in enumerate(used)}
                final_coloring = [remap[c] for c in coloring]
                return [int(c) for c in final_coloring]
            
    return list(range(n))
# EVOLVE-BLOCK-END
