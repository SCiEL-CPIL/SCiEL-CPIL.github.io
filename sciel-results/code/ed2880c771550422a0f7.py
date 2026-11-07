"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 85

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def solve_with_k_colors(k):
        color = [-1] * n
        # adj_pair[(a, b)] = list of lists, adj_pair[(a,b)][v] is list of neighbors in subgraph (a,b)
        adj_pair = {}
        for a in range(k):
            for b in range(a + 1, k):
                adj_pair[(a, b)] = [[] for _ in range(n)]

        def check_connection(pair_key, u, v):
            if u == v:
                return True
            visited = set()
            stack = [u]
            visited.add(u)
            while stack:
                node = stack.pop()
                if node == v:
                    return True
                for nb in adj_pair[pair_key][node]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            return False

        def add_edge(pair_key, u, v):
            adj_pair[pair_key][u].append(v)
            adj_pair[pair_key][v].append(u)

        def remove_edge(pair_key, u, v):
            # Remove first occurrence
            adj_pair[pair_key][u].remove(v)
            adj_pair[pair_key][v].remove(u)

        def backtrack(remaining):
            if remaining == 0:
                return True
            
            best_v = -1
            best_colors = []
            min_count = k + 1
            
            for v in range(n):
                if color[v] != -1:
                    continue
                
                available = []
                for c in range(k):
                    valid = True
                    # Check neighbors
                    for u in adj[v]:
                        if color[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check bicolored cycles
                    conflict = False
                    for u in adj[v]:
                        cu = color[u]
                        if cu != -1 and cu != c:
                            a, b = (c, cu) if c < cu else (cu, c)
                            if check_connection((a, b), v, u):
                                conflict = True
                                break
                    if not conflict:
                        available.append(c)
                
                if len(available) < min_count:
                    min_count = len(available)
                    best_v = v
                    best_colors = available
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        break
            
            if best_v == -1:
                return False
            
            for c in best_colors:
                color[best_v] = c
                added = []
                success = True
                for u in adj[best_v]:
                    cu = color[u]
                    if cu != -1 and cu != c:
                        a, b = (c, cu) if c < cu else (cu, c)
                        added.append((a, b, best_v, u))
                        add_edge((a, b), best_v, u)
                
                if backtrack(remaining - 1):
                    return True
                
                # Undo
                for a, b, u, v in added:
                    remove_edge((a, b), u, v)
                color[best_v] = -1
            
            return False

        return backtrack(n)

    for k in range(3, 6):
        if time.time() > deadline:
            break
        res = solve_with_k_colors(k)
        if res is not None:
            # Verify
            valid = True
            for u, v in edges:
                if res[u] == res[v]:
                    valid = False
                    break
            if valid:
                colors_used = set(res)
                for a in colors_used:
                    for b in colors_used:
                        if a >= b: continue
                        verts = [i for i in range(n) if res[i] in (a, b)]
                        if len(verts) > 1:
                            visited = set()
                            is_forest = True
                            for start in verts:
                                if start not in visited:
                                    queue = [start]
                                    visited.add(start)
                                    ec = 0
                                    vc = 0
                                    while queue:
                                        node = queue.pop()
                                        vc += 1
                                        for nb in adj[node]:
                                            if nb in verts:
                                                ec += 1
                                                if nb not in visited:
                                                    visited.add(nb)
                                                    queue.append(nb)
                                    if ec // 2 != vc - 1:
                                        is_forest = False
                                        break
                            if not is_forest:
                                valid = False
                                break
                    if not valid: break
            if valid:
                return [int(x) for x in res]

    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
