# EVOLVE-BLOCK-START
import sys
from collections import defaultdict
import copy

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute degrees for ordering
    deg = [len(adj[i]) for i in range(n)]
    # Order vertices by descending degree
    order = list(range(n))
    order.sort(key=lambda x: -deg[x])
    
    def solve_with_k(k):
        # DSU for each pair (c1, c2) with c1 < c2
        # We store parent and rank arrays
        # dsu[c1][c2] = [parent, rank]
        # Only for c1 < c2
        
        # Initialize DSUs
        dsu_parent = {}
        dsu_rank = {}
        
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                dsu_parent[(c1, c2)] = list(range(n))
                dsu_rank[(c1, c2)] = [0] * n
                
        def find(pair, x):
            p = dsu_parent[pair]
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x
            
        def union(pair, x, y):
            p = dsu_parent[pair]
            r = dsu_rank[pair]
            rx = find(pair, x)
            ry = find(pair, y)
            if rx == ry:
                return False
            if r[rx] < r[ry]:
                rx, ry = ry, rx
            p[ry] = rx
            if r[rx] == r[ry]:
                r[rx] += 1
            return True
            
        assign = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
                
            v = order[idx]
            
            # Try each color
            for c in range(k):
                # Check if neighbor has same color
                valid = True
                for u in adj[v]:
                    if assign[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign v to c
                assign[v] = c
                
                # Check and update DSUs for all pairs involving c
                conflict = False
                # Store states of DSUs that might be modified
                # We only modify pairs (min(c, c2), max(c, c2))
                # We need to restore them if we backtrack
                # Since we modify in place, we must save the specific changes or copy the whole DSU state for affected pairs
                
                # To be safe and simple, we copy the parent/rank arrays for affected pairs
                # Affected pairs: (c, c2) for all c2 != c
                # Number of pairs: k-1
                # Size of each: n
                # Copy cost: (k-1) * n
                # With k=3, n=50, cost is 100 ints. Very cheap.
                
                saved_parents = {}
                saved_ranks = {}
                for c2 in range(k):
                    if c2 == c:
                        continue
                    pair = (min(c, c2), max(c, c2))
                    saved_parents[pair] = dsu_parent[pair].copy()
                    saved_ranks[pair] = dsu_rank[pair].copy()
                    
                # Perform unions
                for u in adj[v]:
                    cu = assign[u]
                    if cu != -1 and cu != c:
                        pair = (min(c, cu), max(c, cu))
                        # Check if union would create cycle
                        if find(pair, u) == find(pair, v):
                            conflict = True
                            break
                        union(pair, u, v)
                            
                if not conflict:
                    if backtrack(idx + 1):
                        return True
                        
                # Undo: Restore DSUs and assign
                for c2 in range(k):
                    if c2 == c:
                        continue
                    pair = (min(c, c2), max(c, c2))
                    dsu_parent[pair] = saved_parents[pair]
                    dsu_rank[pair] = saved_ranks[pair]
                    
                assign[v] = -1
                
            return False
            
        if backtrack(0):
            return assign
        return None

    # Try increasing number of colors
    for k in range(3, 11):
        res = solve_with_k(k)
        if res is not None:
            return [int(x) for x in res]
            
    # Fallback
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
