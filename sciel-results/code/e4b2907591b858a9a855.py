import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _count_bicolored_cycles(colors, edges, k):
    """
    Returns the number of bicolored cycles.
    Also returns a set of vertices involved in any bicolored cycle.
    """
    total_cycles = 0
    cycle_vertices = set()
    
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Build subgraph induced by c1, c2
            verts = set()
            for i in range(len(colors)):
                if colors[i] == c1 or colors[i] == c2:
                    verts.add(i)
            
            if len(verts) < 3:
                continue
                
            # Use Union-Find to count cycles
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

            cycles_found = 0
            for u, v in edges:
                if u in verts and v in verts:
                    if union(u, v):
                        cycles_found += 1
            
            if cycles_found > 0:
                total_cycles += cycles_found
                cycle_vertices.update(verts)
                
    return total_cycles, cycle_vertices

def _find_bicolored_cycle(colors, c1, c2, adj):
    """
    Find a specific bicolored cycle using DFS.
    Returns a list of vertices in the cycle, or None.
    """
    verts = set()
    for i in range(len(colors)):
        if colors[i] == c1 or colors[i] == c2:
            verts.add(i)
    
    if len(verts) < 3:
        return None
    
    # Build subgraph adjacency
    sub_adj = defaultdict(list)
    for u, v in P['edges']:
        if u in verts and v in verts:
            sub_adj[u].append(v)
            sub_adj[v].append(u)
    
    # DFS to find a cycle
    visited = set()
    parent = {}
    
    def dfs(u, p):
        visited.add(u)
        for v in sub_adj[u]:
            if v == p:
                continue
            if v in visited:
                # Found a cycle
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                return cycle
            parent[v] = u
            res = dfs(v, u)
            if res:
                return res
        return None
    
    for v in verts:
        if v not in visited:
            parent[v] = -1
            res = dfs(v, -1)
            if res:
                return res
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s limit
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try k=3 first, then k=4
    for k in [3, 4]:
        max_restarts = 50 if k == 3 else 20
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random coloring
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Check initial validity
            cost, cycle_verts = _count_bicolored_cycles(colors, edges, k)
            proper = True
            for u, v in edges:
                if colors[u] == colors[v]:
                    proper = False
                    break
            
            if cost == 0 and proper:
                return [int(c) for c in colors]
            
            # Local search: target bicolored cycles
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Check validity
                cost, cycle_verts = _count_bicolored_cycles(colors, edges, k)
                proper = True
                improper_verts = set()
                for u, v in edges:
                    if colors[u] == colors[v]:
                        proper = False
                        improper_verts.add(u)
                        improper_verts.add(v)
                
                if cost == 0 and proper:
                    return [int(c) for c in colors]
                
                # If improper, fix improper vertices first
                if not proper:
                    v = random.choice(list(improper_verts))
                    valid_colors = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                    if valid_colors:
                        colors[v] = random.choice(valid_colors)
                    continue
                
                # Find a specific bicolored cycle
                cycle = None
                for c1 in range(k):
                    for c2 in range(c1 + 1, k):
                        cycle = _find_bicolored_cycle(colors, c1, c2, adj)
                        if cycle:
                            break
                    if cycle:
                        break
                
                if not cycle:
                    # No cycle found, try random move
                    v = random.randint(0, n - 1)
                    valid_colors = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                    if valid_colors:
                        colors[v] = random.choice(valid_colors)
                    continue
                
                # Try to break the cycle by recoloring a vertex in it
                broke = False
                for v in cycle:
                    for new_c in range(k):
                        if new_c == colors[v]:
                            continue
                        if any(colors[nb] == new_c for nb in adj[v]):
                            continue
                        
                        old = colors[v]
                        colors[v] = new_c
                        new_cost, _ = _count_bicolored_cycles(colors, edges, k)
                        
                        # Check proper
                        new_proper = True
                        for nb in adj[v]:
                            if colors[nb] == new_c:
                                new_proper = False
                                break
                        
                        if new_cost < cost and new_proper:
                            broke = True
                            break
                        colors[v] = old
                    
                    if broke:
                        break
                
                if not broke:
                    # Random move
                    v = random.randint(0, n - 1)
                    valid_colors = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                    if valid_colors:
                        colors[v] = random.choice(valid_colors)
    
    # Fallback: return a deterministic 4-coloring
    # This may not be valid, but we need to return something
    return [i % 4 for i in range(n)]
