import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def is_valid(colors):
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Check bicolored cycles
        # For each pair of colors, check if induced subgraph is a forest
        # We only need to check pairs that actually appear in the coloring
        used_colors = set(colors)
        color_list = list(used_colors)
        
        for i in range(len(color_list)):
            for j in range(i+1, len(color_list)):
                c1 = color_list[i]
                c2 = color_list[j]
                
                # Union-Find to detect cycles in the subgraph induced by c1 and c2
                # Vertices are those with color c1 or c2
                # Edges are those connecting a c1 vertex to a c2 vertex
                
                parent = {}
                rank = {}
                
                def find(x):
                    if x not in parent:
                        parent[x] = x
                        rank[x] = 0
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                
                def union(x, y):
                    px = find(x)
                    py = find(y)
                    if px == py:
                        return True # Cycle found
                    if rank[px] < rank[py]:
                        parent[px] = py
                    elif rank[px] > rank[py]:
                        parent[py] = px
                    else:
                        parent[py] = px
                        rank[px] += 1
                    return False

                # Iterate over edges
                for u, v in edges:
                    cu = colors[u]
                    cv = colors[v]
                    if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                        if union(u, v):
                            return False
                            
        return True

    def count_violations(colors):
        violations = 0
        # Proper coloring violations
        for u, v in edges:
            if colors[u] == colors[v]:
                violations += 1
        
        # Bicolored cycle violations
        used_colors = set(colors)
        color_list = list(used_colors)
        
        for i in range(len(color_list)):
            for j in range(i+1, len(color_list)):
                c1 = color_list[i]
                c2 = color_list[j]
                
                parent = {}
                rank = {}
                
                def find(x):
                    if x not in parent:
                        parent[x] = x
                        rank[x] = 0
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                
                def union(x, y):
                    px = find(x)
                    py = find(y)
                    if px == py:
                        return 1
                    if rank[px] < rank[py]:
                        parent[px] = py
                    elif rank[px] > rank[py]:
                        parent[py] = px
                    else:
                        parent[py] = px
                        rank[px] += 1
                    return 0

                for u, v in edges:
                    cu = colors[u]
                    cv = colors[v]
                    if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                        violations += union(u, v)
                        
        return violations

    def solve(k, time_limit=10):
        start_time = time.time()
        best_colors = None
        best_violations = float('inf')
        
        while time.time() - start_time < time_limit:
            # Random initialization
            colors = [random.randint(0, k-1) for _ in range(n)]
            violations = count_violations(colors)
            
            if violations == 0:
                return colors
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
            
            # Local search
            T = 1.0
            for _ in range(1000):
                if time.time() - start_time > time_limit:
                    break
                
                v = random.randint(0, n-1)
                old_c = colors[v]
                new_c = random.randint(0, k-1)
                if new_c == old_c:
                    continue
                
                colors[v] = new_c
                new_violations = count_violations(colors)
                delta = new_violations - violations
                
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    violations = new_violations
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                    if violations == 0:
                        return colors
                else:
                    colors[v] = old_c
                
                T *= 0.999 # Cooling
                
        return best_colors

    # Try k=3 first, then k=4
    # If 3 fails quickly, move to 4
    res3 = solve(3, 5)
    if res3 is not None and is_valid(res3):
        return res3
        
    res4 = solve(4, 20)
    if res4 is not None and is_valid(res4):
        return res4
        
    # Fallback to k=5
    res5 = solve(5, 20)
    if res5 is not None and is_valid(res5):
        return res5
        
    # If all fail, return a proper 4-coloring (might have cycles, but better than invalid?)
    # The task says "Any broken rule makes the whole result invalid".
    # So we must return a valid one.
    # If search fails, we can try a deterministic 4-coloring heuristic.
    # But search should work.
    
    # Last resort: return 50 colors (valid but score 1.0)
    return list(range(n))
# EVOLVE-BLOCK-END
