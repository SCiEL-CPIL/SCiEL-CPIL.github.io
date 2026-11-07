"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute edge list for fast access in checks
    # We need to check if a specific pair of colors forms a cycle.
    # This is done by building a DSU on the subgraph induced by those two colors.
    
    def check_bicolored_forest(colors, n, edges):
        """
        Check if the coloring is valid:
        1. Adjacent vertices have different colors.
        2. For every pair of colors (c1, c2), the subgraph induced by vertices 
           with colors c1 or c2 is a forest.
        """
        # Check adjacent vertices
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Group edges by color pair
        pair_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if cu < cv:
                pair_edges[(cu, cv)].append((u, v))
            elif cv < cu:
                pair_edges[(cv, cu)].append((u, v))
            
        # Check each pair for cycles using Union-Find
        for pair, elist in pair_edges.items():
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
                    return False # Cycle found
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
                
            for u, v in elist:
                if not union(u, v):
                    return False
        return True

    def backtrack_search(num_colors, deadline, rng):
        """
        Backtracking search with constraint propagation.
        """
        colors = [-1] * n
        
        # Helper to check if assigning 'color' to 'vertex' creates a bicolored cycle
        # We only need to check pairs (color, c_nb) for each neighbor nb with color c_nb.
        def is_valid(vertex, color):
            # 1. Check neighbor color conflict
            for nb in adj[vertex]:
                if colors[nb] == color:
                    return False
            
            # 2. Check bicolored cycle creation
            # For each neighbor with a distinct color, check if adding this vertex 
            # to the (color, colors[nb]) subgraph creates a cycle.
            for nb in adj[vertex]:
                nb_color = colors[nb]
                if nb_color == -1 or nb_color == color:
                    continue
                
                # We need to check if the subgraph induced by colors {color, nb_color} 
                # plus the edge (vertex, nb) creates a cycle.
                # This is equivalent to checking if vertex and nb are already connected 
                # in the subgraph induced by {color, nb_color} excluding the edge (vertex, nb) 
                # but including all other edges and vertices with these colors.
                
                # Efficient check: 
                # Build DSU on all vertices with colors in {color, nb_color}.
                # If vertex and nb are already connected, adding the edge creates a cycle.
                
                # To avoid rebuilding DSU from scratch for every neighbor, 
                # we can just check connectivity in the current partial assignment.
                # However, since 'vertex' is not yet assigned, we treat it as having 'color' 
                # for the purpose of this check.
                
                # Let's collect all edges between vertices with colors in {color, nb_color}.
                # Note: 'vertex' is assigned 'color' hypothetically.
                
                # Optimization: Only check if the number of such edges is small or use a global DSU?
                # Given n=50, rebuilding a small DSU is fast enough.
                
                # Identify vertices involved:
                # All v such that colors[v] in {color, nb_color} OR v == vertex.
                
                # Let's just run a BFS/DFS from 'nb' in the subgraph induced by {color, nb_color}
                # to see if we can reach 'vertex'.
                # Wait, 'vertex' is not in the subgraph yet.
                # The edge (vertex, nb) connects them.
                # A cycle exists if there is a path from vertex to nb in the subgraph 
                # induced by {color, nb_color} that does NOT use the edge (vertex, nb).
                # But 'vertex' is only connected to its neighbors.
                # Actually, simpler: 
                # In the subgraph induced by colors {color, nb_color}, if we add vertex (with color 'color'),
                # does it create a cycle?
                # Yes, if vertex is connected to nb via other paths in this subgraph.
                
                # Let's do a BFS from nb in the subgraph of colors {color, nb_color}.
                # We treat 'vertex' as color 'color'.
                visited = set()
                queue = [nb]
                visited.add(nb)
                found_cycle = False
                
                while queue:
                    curr = queue.pop(0)
                    if curr == vertex:
                        # We found a path from nb to vertex. 
                        # Since we are adding edge (vertex, nb), this forms a cycle.
                        found_cycle = True
                        break
                    for next_v in adj[curr]:
                        if next_v in visited:
                            continue
                        # Check if next_v is in the subgraph
                        # next_v is in subgraph if its color is in {color, nb_color}
                        # Or if next_v is vertex (which we are hypothetically assigning)
                        if next_v == vertex:
                            # We are moving towards vertex. 
                            # If we reach vertex, we have a path.
                            # But wait, we start at nb. If we reach vertex, 
                            # it means there is a path nb -> ... -> vertex.
                            # Adding edge (vertex, nb) closes the loop.
                            # So yes, cycle.
                            visited.add(next_v)
                            queue.append(next_v)
                        else:
                            c_next = colors[next_v]
                            if c_next == color or c_next == nb_color:
                                visited.add(next_v)
                                queue.append(next_v)
                                
                if found_cycle:
                    return False
            
            return True

        def solve(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            # Heuristic: Try colors in a fixed order or random?
            # Random can help escape local minima, but for CSP, deterministic order is often fine.
            # Let's try all colors.
            for c in range(num_colors):
                if is_valid(idx, c):
                    colors[idx] = c
                    if solve(idx + 1):
                        return True
                    colors[idx] = -1
            return False

        # Shuffle vertex order? No, fixed order is standard for backtracking.
        # We can try to pick the next vertex with fewest available colors (MRV).
        # For simplicity and speed, let's stick to index order first. 
        # If 3 colors fail, we might need MRV.
        
        # Let's implement a simple MRV selection to improve performance.
        # Instead of fixed idx, pick the unassigned vertex with the fewest valid colors.
        
        def solve_mrv():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with minimum valid colors
            min_count = num_colors + 1
            min_vertex = -1
            valid_colors_list = []
            
            for i in range(n):
                if colors[i] != -1:
                    continue
                count = 0
                vcols = []
                for c in range(num_colors):
                    if is_valid(i, c):
                        count += 1
                        vcols.append(c)
                if count < min_count:
                    min_count = count
                    min_vertex = i
                    valid_colors_list = vcols
                if count == 0:
                    return False # Dead end
            
            if min_vertex == -1:
                # All assigned
                return True
                
            # Try valid colors for min_vertex
            # Shuffle for randomness?
            rng.shuffle(valid_colors_list)
            for c in valid_colors_list:
                colors[min_vertex] = c
                if solve_mrv():
                    return True
                colors[min_vertex] = -1
            return False

        # Try MRV solver
        if solve_mrv():
            return colors
        return None

    deadline = time.time() + 85  # 5s safety margin
    
    rng = random.Random(42)
    
    # Try 3 colors first
    # If 3 colors is possible, score is 50/3 = 16.67
    result = backtrack_search(3, deadline, rng)
    if result:
        # Verify
        if check_bicolored_forest(result, n, edges):
            return [int(c) for c in result]
            
    # Try 4 colors
    # If 4 colors is possible, score is 50/4 = 12.5
    result = backtrack_search(4, deadline, rng)
    if result:
        if check_bicolored_forest(result, n, edges):
            return [int(c) for c in result]
            
    # Fallback: Greedy or Random Local Search for 5+ colors if backtracking failed
    # But backtracking with MRV should be very strong.
    # Let's add a quick local search for 5 colors just in case.
    
    def local_search(num_colors, iterations, deadline):
        colors = [rng.randint(0, num_colors - 1) for _ in range(n)]
        for _ in range(iterations):
            if time.time() > deadline:
                break
            if check_bicolored_forest(colors, n, edges):
                return colors
            # Pick a random vertex and change its color
            i = rng.randint(0, n - 1)
            old_c = colors[i]
            new_c = rng.randint(0, num_colors - 1)
            if new_c == old_c:
                continue
            colors[i] = new_c
            # We don't revert immediately, we keep the change if it doesn't immediately break 
            # basic adjacency? No, the constraint is global.
            # Simple hill climbing: accept if valid, else revert.
            if not check_bicolored_forest(colors, n, edges):
                colors[i] = old_c
                
        return None

    # Try 5 colors with local search
    result = local_search(5, 1000, deadline)
    if result:
        return [int(c) for c in result]
        
    # Final fallback: All distinct colors (always valid, score 1.0)
    return list(range(n))

# EVOLVE-BLOCK-END
