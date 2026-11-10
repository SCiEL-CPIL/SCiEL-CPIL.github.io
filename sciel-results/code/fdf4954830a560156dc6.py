import time
import random

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def find_bicolored_cycle(colors):
        # Check all pairs of colors
        colors_used = set(colors)
        color_list = list(colors_used)
        for i in range(len(color_list)):
            for j in range(i+1, len(color_list)):
                c1, c2 = color_list[i], color_list[j]
                # Induced subgraph on c1, c2
                # Check for cycle
                # Use DFS
                visited = {False} * n
                parent = [-1] * n
                
                # We only care about vertices with color c1 or c2
                # Start DFS from any unvisited vertex with color c1 or c2
                for start in range(n):
                    if colors[start] not in (c1, c2):
                        continue
                    if visited[start]:
                        continue
                    
                    stack = [(start, -1)]
                    visited[start] = True
                    while stack:
                        u, p = stack.pop()
                        for v in adj[u]:
                            if colors[v] not in (c1, c2):
                                continue
                            if not visited[v]:
                                visited[v] = True
                                parent[v] = u
                                stack.append((v, u))
                            elif v != p:
                                # Found cycle
                                # Reconstruct cycle
                                cycle = []
                                curr = u
                                while curr != v:
                                    cycle.append(curr)
                                    curr = parent[curr]
                                cycle.append(v)
                                return cycle
        return None

    def is_valid(colors):
        # Proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # No bicolored cycles
        if find_bicolored_cycle(colors) is not None:
            return False
        return True

    def solve():
        # Try a few initial configurations
        # 1. i % 3
        # 2. Random
        # 3. Greedy repair
        
        for attempt in range(1000):
            if time.time() > deadline:
                return None
            
            if attempt == 0:
                colors = [i % 3 for i in range(n)]
            else:
                # Random 3-coloring
                colors = [random.randint(0, 2) for _ in range(n)]
                
            # Repair loop
            max_repair_steps = 100
            for step in range(max_repair_steps):
                if time.time() > deadline:
                    return None
                
                # Check properness
                proper = True
                for u, v in edges:
                    if colors[u] == colors[v]:
                        proper = False
                        break
                
                if not proper:
                    # Fix properness: pick a conflict and change one
                    # Find first conflict
                    for u, v in edges:
                        if colors[u] == colors[v]:
                            # Change v to a different color
                            for c in range(3):
                                if c != colors[u]:
                                    # Check if c conflicts with other neighbors of v
                                    ok = True
                                    for nb in adj[v]:
                                        if colors[nb] == c:
                                            ok = False
                                            break
                                    if ok:
                                        colors[v] = c
                                        break
                                    else:
                                        # If no simple fix, try changing u
                                        for c in range(3):
                                            if c != colors[v]:
                                                ok = True
                                                for nb in adj[u]:
                                                    if colors[nb] == c:
                                                        ok = False
                                                        break
                                                if ok:
                                                    colors[u] = c
                                                    break
                                        break
                            break
                    continue
                
                # Check cycles
                cycle = find_bicolored_cycle(colors)
                if cycle is None:
                    return colors
                
                # Break cycle
                # Pick a random vertex in cycle
                v = random.choice(cycle)
                # Determine colors in cycle
                cyc_colors = set(colors[x] for x in cycle)
                # The 3rd color
                all_colors = {0, 1, 2}
                third_color = (all_colors - cyc_colors).pop()
                
                # Can we recolor v to third_color?
                ok = True
                for nb in adj[v]:
                    if colors[nb] == third_color:
                        ok = False
                        break
                
                if ok:
                    colors[v] = third_color
                else:
                    # Try another vertex in cycle
                    success = False
                    for v2 in cycle:
                        if v2 == v:
                            continue
                        ok2 = True
                        for nb in adj[v2]:
                            if colors[nb] == third_color:
                                ok2 = False
                                break
                        if ok2:
                            colors[v2] = third_color
                            success = True
                            break
                    if not success:
                        # Stuck, break out to restart
                        break
                        
        return None

    result = solve()
    if result is None:
        # Fallback: return a valid coloring with more colors if possible?
        # Or just return the seed?
        # But we need to minimize.
        # If 3 colors fail, try 4?
        # For now, if 3 fails, we might return a random valid one with 4 colors.
        # Let's add a fallback for 4 colors.
        pass
        
    # If solve failed, try a simple 4-coloring fallback
    # Pattern: i % 4
    # Check if valid. If not, random 4-coloring repair.
    
    if result is None:
        # Try 4 colors
        for attempt in range(1000):
            if time.time() > deadline:
                return list(range(n)) # Fallback to seed-like (invalid but returns)
            
            if attempt == 0:
                colors = [i % 4 for i in range(n)]
            else:
                colors = [random.randint(0, 3) for _ in range(n)]
                
            max_repair_steps = 100
            for step in range(max_repair_steps):
                if time.time() > deadline:
                    return list(range(n))
                
                proper = True
                for u, v in edges:
                    if colors[u] == colors[v]:
                        proper = False
                        break
                
                if not proper:
                    for u, v in edges:
                        if colors[u] == colors[v]:
                            for c in range(4):
                                if c != colors[u]:
                                    ok = True
                                    for nb in adj[v]:
                                        if colors[nb] == c:
                                            ok = False
                                            break
                                    if ok:
                                        colors[v] = c
                                        break
                            break
                    continue
                
                cycle = find_bicolored_cycle(colors)
                if cycle is None:
                    return colors
                
                v = random.choice(cycle)
                cyc_colors = set(colors[x] for x in cycle)
                all_colors = {0, 1, 2, 3}
                available = list(all_colors - cyc_colors)
                if not available:
                    break
                third_color = random.choice(available)
                
                ok = True
                for nb in adj[v]:
                    if colors[nb] == third_color:
                        ok = False
                        break
                if ok:
                    colors[v] = third_color
                else:
                    success = False
                    for v2 in cycle:
                        if v2 == v: continue
                        ok2 = True
                        for nb in adj[v2]:
                            if colors[nb] == third_color:
                                ok2 = False
                                break
                        if ok2:
                            colors[v2] = third_color
                            success = True
                            break
                    if not success:
                        break
                        
    return list(range(n)) # Final fallback

# EVOLVE-BLOCK-END
