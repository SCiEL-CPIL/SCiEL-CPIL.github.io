import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest_optimized(colors, edges, adj, n, v, c, assigned):
    """
    Check if assigning color c to v creates a bicolored cycle.
    Only checks pairs (c, c') where c' is used in assigned vertices.
    """
    for c_prime in range(len(set(colors[i] for i in range(n) if assigned[i] and colors[i] != -1))):
        # Actually, just iterate 0..k-1 is fine, k is small
        pass # Placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list for fast access
    edge_set = set(map(tuple, edges))
    
    def _has_cycle_in_subgraph(colors, assigned, c1, c2, exclude_v=None):
        """
        Check if subgraph induced by c1, c2 (excluding exclude_v) has a cycle.
        Returns True if it has a cycle.
        """
        if exclude_v is not None:
            # We are checking if adding exclude_v would create a cycle.
            # This is equivalent to checking if the subgraph (excluding exclude_v) has a path between any two neighbors of exclude_v that are in the subgraph.
            # Or simpler: just check if the subgraph (excluding exclude_v) is a forest.
            # If it is a forest, adding exclude_v and its edges might create a cycle.
            # Actually, the standard check is:
            # 1. Identify neighbors of exclude_v that have color c1 or c2.
            # 2. Check if any two of these neighbors are in the same connected component of G[c1, c2] \ {exclude_v}.
            # 3. If yes, adding exclude_v creates a cycle.
            
            neighbors = adj[exclude_v]
            relevant_neighbors = []
            for nb in neighbors:
                if assigned[nb] and (colors[nb] == c1 or colors[nb] == c2):
                    relevant_neighbors.append(nb)
            
            if len(relevant_neighbors) <= 1:
                return False # No cycle can be formed by adding a leaf or single edge to a forest
                
            # Build Union-Find for G[c1, c2] \ {exclude_v}
            verts = set()
            for i in range(n):
                if i != exclude_v and assigned[i] and (colors[i] == c1 or colors[i] == c2):
                    verts.add(i)
            
            if not verts:
                return False
                
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
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
                
            for u, v in edges:
                if u != exclude_v and v != exclude_v:
                    if u in verts and v in verts:
                        union(u, v)
            
            # Check if any two relevant neighbors are in the same component
            roots = set()
            for nb in relevant_neighbors:
                r = find(nb)
                if r in roots:
                    return True
                roots.add(r)
            return False

        else:
            # Standard check: is G[c1, c2] a forest?
            verts = set()
            for i in range(n):
                if assigned[i] and (colors[i] == c1 or colors[i] == c2):
                    verts.add(i)
            
            if len(verts) <= 1:
                return False
                
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
                
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True # Cycle found
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return False
                
            for u, v in edges:
                if u in verts and v in verts:
                    if union(u, v):
                        return True
            return False

    # Backtracking function
    def backtrack(idx, k):
        if time.time() > deadline - 2:
            return None
        
        if idx == n:
            return colors[:]
        
        v = idx
        
        # Try all colors
        for c in range(k):
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if assigned[nb] and colors[nb] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check bicolored forest
            # Assign c to v
            colors[v] = c
            assigned[v] = True
            
            conflict = False
            # Check all pairs involving c
            # We only need to check if adding v creates a cycle in G[c, c']
            # This is handled by _has_cycle_in_subgraph with exclude_v=v
            for c_prime in range(k):
                if c_prime == c:
                    continue
                # Check if c_prime is used
                if any(assigned[i] and colors[i] == c_prime for i in range(n)):
                    if _has_cycle_in_subgraph(colors, assigned, c, c_prime, exclude_v=v):
                        conflict = True
                        break
            
            if not conflict:
                res = backtrack(idx + 1, k)
                if res is not None:
                    return res
            else:
                # Revert
                colors[v] = -1
                assigned[v] = False
                continue
            
            # Revert
            colors[v] = -1
            assigned[v] = False
            
        return None

    # Try k=3, 4, 5
    for k in [3, 4, 5]:
        colors = [-1] * n
        assigned = [False] * n
        res = backtrack(0, k)
        if res is not None:
            return [int(c) for c in res]
            
    # Fallback
    colors = [None] * n
    for v in range(n):
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] is not None:
                forbidden.add(colors[nb])
        for c in range(10):
            if c not in forbidden:
                colors[v] = c
                break
        if colors[v] is None:
            colors[v] = 0
    return [int(c) for c in colors]
