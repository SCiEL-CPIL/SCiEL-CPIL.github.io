"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def verify(color):
        if not isinstance(color, (list, tuple)):
            return False
        if len(color) != n:
            return False
        for c in color:
            if not isinstance(c, int) or isinstance(c, bool):
                return False
            if c < 0 or c > 49:
                return False
        for u, v in edges:
            if color[u] == color[v]:
                return False
        
        colors_used = set(color)
        for a in colors_used:
            for b in colors_used:
                if a >= b:
                    continue
                verts = [v for v in range(n) if color[v] in (a, b)]
                if len(verts) > 1:
                    visited = set()
                    is_tree = True
                    for start in verts:
                        if start not in visited:
                            queue = deque([start])
                            visited.add(start)
                            ec = 0
                            vc = 0
                            while queue:
                                node = queue.popleft()
                                vc += 1
                                for nb in adj[node]:
                                    if color[nb] in (a, b):
                                        ec += 1
                                        if nb not in visited:
                                            visited.add(nb)
                                            queue.append(nb)
                            if ec // 2 != vc - 1:
                                is_tree = False
                                break
                    if not is_tree:
                        return False
        return True

    def greedy_color(k):
        color = [-1] * n
        for v in range(n):
            used = set()
            for u in adj[v]:
                if color[u] >= 0:
                    used.add(color[u])
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            else:
                return None
        if verify(color):
            return [int(c) for c in color]
        return None

    def backtrack_color(k, time_limit):
        color = [-1] * n
        # DSU for pairs
        dsu = {}
        for a in range(k):
            for b in range(a + 1, k):
                dsu[(a, b)] = list(range(n))
        
        def find(parent, x):
            while parent[x] != x:
                x = parent[x]
            return x

        def union(parent, a, b):
            ra = find(parent, a)
            rb = find(parent, b)
            if ra == rb:
                return False
            parent[ra] = rb
            return True

        union_log = []

        def do_union(pair_key, u, v):
            parent = dsu[pair_key]
            ra = find(parent, u)
            rb = find(parent, v)
            if ra == rb:
                return False
            parent[ra] = rb
            union_log.append((pair_key, ra, rb))
            return True

        def undo_unions(count):
            for _ in range(count):
                pair_key, ra, rb = union_log.pop()
                dsu[pair_key][ra] = ra

        def get_legal_colors(v):
            legal = []
            for c in range(k):
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                conflict = False
                for u in adj[v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        a, b = (c, cu) if c < cu else (cu, c)
                        pair_key = (a, b)
                        parent = dsu[pair_key]
                        if find(parent, v) == find(parent, u):
                            conflict = True
                            break
                if not conflict:
                    legal.append(c)
            return legal

        def bt():
            if time.time() > time_limit:
                return False
            
            best_v = -1
            best_legal = None
            min_count = k + 1
            
            for v in range(n):
                if color[v] == -1:
                    legal = get_legal_colors(v)
                    if len(legal) < min_count:
                        min_count = len(legal)
                        best_v = v
                        best_legal = legal
                        if min_count == 0:
                            return False
                        if min_count == 1:
                            break
            
            if best_v == -1:
                return True

            for c in best_legal:
                color[best_v] = c
                unions_done = 0
                conflict = False
                for u in adj[best_v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        a, b = (c, cu) if c < cu else (cu, c)
                        pair_key = (a, b)
                        if do_union(pair_key, best_v, u):
                            unions_done += 1
                        else:
                            conflict = True
                            break
                
                if conflict:
                    undo_unions(unions_done)
                    color[best_v] = -1
                    continue

                if bt():
                    return True

                undo_unions(unions_done)
                color[best_v] = -1
            
            return False

        if bt():
            return [int(c) for c in color]
        return None

    # Try k=3
    res = greedy_color(3)
    if res:
        return res
    
    res = backtrack_color(3, start_time + 40)
    if res:
        return res

    # Try k=4
    res = greedy_color(4)
    if res:
        return res
    
    res = backtrack_color(4, deadline)
    if res:
        return res

    # Fallback
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
