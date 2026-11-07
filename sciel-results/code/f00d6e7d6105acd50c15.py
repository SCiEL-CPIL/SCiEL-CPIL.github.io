"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    
    # For each color pair, we track edges to detect cycles via union-find
    # A bicolored cycle exists if adding an edge between two vertices of the same color pair
    # creates a cycle in that pair's subgraph
    
    # Strategy: try k=2,3,4,... colors. For each k, try to assign colors greedily
    # with backtracking if needed, checking bicolored cycle constraint.
    
    # First, let's try a simple approach: use k colors and assign greedily
    # For a given k, we need to ensure:
    # 1. Adjacent vertices have different colors
    # 2. For any two colors c1, c2, the subgraph induced by vertices with colors c1 or c2 is a forest
    
    # The key insight: if we use k colors, each color pair (c1,c2) can have at most
    # (number of vertices with color c1 or c2) - 1 edges without creating a cycle.
    
    # Let's try a constructive approach for small k:
    # For k=4: use a 4-coloring based on (i mod 2, j mod 2) for a grid-like structure
    
    # Actually, let's try to find a valid coloring with few colors using a greedy
    # approach with careful ordering.
    
    # Order vertices by degree (descending) for better greedy performance
    degrees = {v: len(adj[v]) for v in range(n)}
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    def check_bicolored_forest(coloring, max_color):
        """Check if the coloring is valid: no bicolored cycles."""
        # For each pair of colors, build the subgraph and check for cycles
        for c1 in range(max_color + 1):
            for c2 in range(c1 + 1, max_color + 1):
                # Get all vertices with color c1 or c2
                vertices = [v for v in range(n) if coloring[v] in (c1, c2)]
                if len(vertices) <= 1:
                    continue
                # Build subgraph edges
                vset = set(vertices)
                sub_edges = [(u, v) for u, v in edges if u in vset and v in vset]
                # Check for cycle using union-find
                parent = {v: v for v in vertices}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False  # cycle detected
                    parent[rx] = ry
                    return True
                for u, v in sub_edges:
                    if not union(u, v):
                        return False  # cycle found
        return True
    
    def is_valid_coloring(coloring, max_color):
        """Check all constraints."""
        # Check adjacent vertices differ
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check bicolored forest constraint
        return check_bicolored_forest(coloring, max_color)
    
    # Try k=2 first
    for k in range(2, 50):
        if time.time() > deadline:
            break
        # Try greedy assignment with k colors
        # Use multiple random orderings
        import random
        random.seed(42)
        
        best_coloring = None
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Create a random ordering
            order_random = list(range(n))
            random.shuffle(order_random)
            
            coloring = [-1] * n
            valid = True
            for v in order_random:
                # Try each color
                assigned = False
                for c in range(k):
                    # Check if color c is valid for vertex v
                    ok = True
                    # Check neighbors already colored
                    for nb in adj[v]:
                        if coloring[nb] == c:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Tentatively assign and check bicolored cycle constraint
                    coloring[v] = c
                    # Check if this creates a bicolored cycle
                    # Only need to check pairs involving color c and existing colors
                    cycle_found = False
                    for c2 in range(k):
                        if c2 == c:
                            continue
                        # Check pair (c, c2)
                        # Get vertices with color c or c2
                        vset = set()
                        for u in range(n):
                            if coloring[u] in (c, c2):
                                vset.add(u)
                        if len(vset) <= 1:
                            continue
                        sub_edges = [(u, v) for u, v in edges if u in vset and v in vset]
                        # Union-find
                        parent = {x: x for x in vset}
                        def find(x):
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x
                        def union(x, y):
                            rx, ry = find(x), find(y)
                            if rx == ry:
                                return False
                            parent[rx] = ry
                            return True
                        for u, v in sub_edges:
                            if not union(u, v):
                                cycle_found = True
                                break
                        if cycle_found:
                            break
                    if not cycle_found:
                        assigned = True
                        break
                    else:
                        coloring[v] = -1
                if not assigned:
                    valid = False
                    break
            if valid:
                best_coloring = coloring[:]
                break
        
        if best_coloring is not None:
            # Verify
            if is_valid_coloring(best_coloring, k - 1):
                # Return as list of Python ints
                result = [int(best_coloring[v]) for v in range(n)]
                return result
    
    # Fallback: use the trivial solution
    return list(range(n))
# EVOLVE-BLOCK-END
