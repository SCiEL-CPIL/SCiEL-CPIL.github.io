"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def is_forest(c1, c2, colors):
        """Check if subgraph induced by colors c1 and c2 is a forest using Union-Find."""
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
        
        for u, v in edges:
            if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                if not union(u, v):
                    return False
        return True
    
    def count_violations(colors, k):
        """Count number of bicolored cycle violations."""
        count = 0
        for i in range(k):
            for j in range(i+1, k):
                if not is_forest(i, j, colors):
                    count += 1
        return count
    
    def backtrack(k, idx, colors, deadline):
        """
        Backtracking search for k-coloring with forward checking.
        Returns True if valid coloring found.
        """
        if time.time() > deadline:
            return False
            
        if idx == n:
            # Final verification
            for i in range(k):
                for j in range(i+1, k):
                    if not is_forest(i, j, colors):
                        return False
            return True
        
        v = order[idx]
        
        # Try colors in order of least constraining
        for c in range(k):
            # 1. Proper coloring check
            valid = True
            for nb in adj[v]:
                if colors[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # 2. Tentative assignment
            colors[v] = c
            
            # 3. Check bicolored forest constraint incrementally
            cycle_created = False
            for c_prime in range(k):
                if c_prime == c:
                    continue
                if not is_forest(c, c_prime, colors):
                    cycle_created = True
                    break
            
            if not cycle_created:
                if backtrack(k, idx + 1, colors, deadline):
                    return True
            
            colors[v] = -1
            
        return False
    
    # Vertex ordering: use a heuristic that considers both degree and structural position
    # Try multiple orderings
    orderings = []
    
    # Ordering 1: descending degree
    degrees = [len(adj[i]) for i in range(n)]
    order1 = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    orderings.append(order1)
    
    # Ordering 2: BFS from vertex 0
    visited = [False] * n
    visited[0] = True
    queue = [0]
    order2 = []
    while queue:
        v = queue.pop(0)
        order2.append(v)
        for nb in adj[v]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    orderings.append(order2)
    
    # Ordering 3: DFS from vertex 0
    visited = [False] * n
    order3 = []
    def dfs(v):
        visited[v] = True
        order3.append(v)
        for nb in adj[v]:
            if not visited[nb]:
                dfs(nb)
    dfs(0)
    orderings.append(order3)
    
    # Try 3 colors with different orderings
    for order in orderings:
        if time.time() > deadline - 30:
            break
        colors = [-1] * n
        if backtrack(3, 0, colors, deadline - 20):
            return [int(c) for c in colors]
    
    # Try 4 colors with different orderings
    for order in orderings:
        if time.time() > deadline - 15:
            break
        colors = [-1] * n
        if backtrack(4, 0, colors, deadline - 10):
            return [int(c) for c in colors]
    
    # If backtracking fails, use a smarter local search
    # Start with a good initial coloring using a pattern
    # The graph has a regular structure, so try periodic patterns
    
    best_colors = None
    best_viol = float('inf')
    
    # Try several structured initial colorings
    patterns = []
    
    # Pattern: (i // 5 + i % 5) % 4
    patterns.append([(i // 5 + i % 5) % 4 for i in range(n)])
    
    # Pattern: (i // 7 + i % 7) % 4  
    patterns.append([(i // 7 + i % 7) % 4 for i in range(n)])
    
    # Pattern: (i // 10 + i % 10) % 4
    patterns.append([(i // 10 + i % 10) % 4 for i in range(n)])
    
    # Pattern: (i * 7) % 4
    patterns.append([(i * 7) % 4 for i in range(n)])
    
    # Pattern: (i * 3 + i // 10) % 4
    patterns.append([(i * 3 + i // 10) % 4 for i in range(n)])
    
    for pattern in patterns:
        # Fix proper coloring
        for _ in range(200):
            proper = True
            for u, v in edges:
                if pattern[u] == pattern[v]:
                    proper = False
                    break
            if proper:
                break
            # Fix a random conflict
            conflicts = [(u, v) for u, v in edges if pattern[u] == pattern[v]]
            if not conflicts:
                break
            u, v = random.choice(conflicts)
            forbidden = set(pattern[nb] for nb in adj[v])
            for c in range(4):
                if c not in forbidden:
                    pattern[v] = c
                    break
        
        viol = count_violations(pattern, 4)
        if viol < best_viol:
            best_viol = viol
            best_colors = pattern[:]
        
        if best_viol == 0:
            break
    
    if best_viol == 0:
        return [int(c) for c in best_colors]
    
    # Simulated annealing to refine
    if best_colors is None:
        best_colors = [random.randint(0, 3) for _ in range(n)]
        # Fix proper coloring
        for _ in range(500):
            proper = True
            for u, v in edges:
                if best_colors[u] == best_colors[v]:
                    proper = False
                    break
            if proper:
                break
            conflicts = [(u, v) for u, v in edges if best_colors[u] == best_colors[v]]
            if not conflicts:
                break
            u, v = random.choice(conflicts)
            forbidden = set(best_colors[nb] for nb in adj[v])
            for c in range(4):
                if c not in forbidden:
                    best_colors[v] = c
                    break
    
    colors = best_colors[:]
    curr_viol = best_viol
    
    T = 2.0
    cooling = 0.999
    for step in range(100000):
        if time.time() > deadline - 3:
            break
        if curr_viol == 0:
            break
            
        # Pick a random vertex
        v = random.randint(0, n-1)
        old_c = colors[v]
        
        # Try a new color
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Check proper coloring
        if any(colors[nb] == new_c for nb in adj[v]):
            continue
            
        # Apply change
        colors[v] = new_c
        new_viol = count_violations(colors, 4)
        
        # Accept or reject
        if new_viol < curr_viol:
            curr_viol = new_viol
            if curr_viol < best_viol:
                best_viol = curr_viol
                best_colors = colors[:]
        else:
            # Metropolis acceptance
            delta = new_viol - curr_viol
            if random.random() < math.exp(-delta / T):
                curr_viol = new_viol
            else:
                colors[v] = old_c
        
        T *= cooling
    
    colors = best_colors
    
    # Final verification
    if count_violations(colors, 4) == 0:
        return [int(c) for c in colors]
    
    # If still invalid, try 5 colors
    colors = [-1] * n
    if backtrack(5, 0, colors, deadline - 2):
        return [int(c) for c in colors]
    
    # Last resort: use all distinct colors (always valid)
    colors = list(range(n))
    return [int(c) for c in colors]

# EVOLVE-BLOCK-END
