"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_all_cycles_biconnected(n, edges):
    """Find all cycles in the graph using a simple DFS-based approach for small graphs.
    We'll find cycles by looking at biconnected components and then enumerating cycles within them."""
    adj = _build_adj()
    cycles = []
    
    # For a 4-regular graph on 50 vertices, we can find cycles by 
    # checking all pairs of vertices and finding paths between them
    # But this is expensive. Instead, let's use the fact that we need
    # to check if a coloring creates a bicolored cycle.
    
    # Alternative: enumerate all simple cycles up to a reasonable length
    # For a 4-regular graph, cycles can be long, but we can use 
    # a different approach: for each pair of colors, check if the 
    # subgraph is a forest using union-find.
    
    # Actually, the most efficient approach is to use the union-find 
    # check during search rather than precomputing all cycles.
    
    return cycles

def _check_valid_coloring(coloring, n, edges):
    """Check if a coloring is valid: no adjacent same color, no bicolored cycles."""
    # Check adjacent vertices have different colors
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    # Check no bicolored cycles using union-find
    # For each pair of colors, check if the subgraph induced by those two colors is a forest
    colors_used = set(coloring)
    color_pairs = list(itertools.combinations(colors_used, 2))
    
    for c1, c2 in color_pairs:
        # Build subgraph with only vertices colored c1 or c2
        # and only edges between such vertices
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
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False  # Found a bicolored cycle
    
    return True

def _greedy_coloring(n, edges, k):
    """Try to find a valid coloring with at most k colors using greedy + backtracking."""
    adj = _build_adj()
    coloring = [-1] * n
    
    # Order vertices by degree (all are 4, so use natural order)
    # Try to assign colors greedily
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = idx
        # Try each color
        for c in range(k):
            # Check if color c is valid for vertex v
            valid = True
            for u in adj[v]:
                if u < idx and coloring[u] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            coloring[v] = c
            
            # Check if this creates a bicolored cycle
            # Only need to check cycles involving v
            # A bicolored cycle involving v must use v and have exactly two colors
            # The colors used in the cycle must include coloring[v]
            
            # For efficiency, we can check: for each color c2 != c,
            # check if there's a cycle in the subgraph induced by {c, c2} 
            # that includes v
            # This is expensive, so we'll do a full check at the end 
            # or use a lighter check
            
            if backtrack(idx + 1):
                return True
            
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
        # Verify the coloring is valid
        if _check_valid_coloring(coloring, n, edges):
            return coloring
    return None

def _mip_search(n, edges, max_k=10, time_limit=60):
    """Use a systematic search to find the minimum number of colors."""
    start_time = time.time()
    adj = _build_adj()
    
    # Try increasing k
    for k in range(2, max_k + 1):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try to find a valid k-coloring
        # Use backtracking with constraint propagation
        
        coloring = [-1] * n
        color_to_vertices = [set() for _ in range(k)]
        
        # Precompute: for each vertex, which colors are forbidden by neighbors
        # This is dynamic, so we'll compute it during search
        
        def is_bicolored_cycle_with_color(c1, c2):
            """Check if adding an edge between two vertices of colors c1 and c2
            creates a bicolored cycle."""
            # This is called during search, so we need an efficient check
            # We'll use union-find on the subgraph induced by colors c1 and c2
            # But we only have partial coloring, so this is tricky
            
            # Alternative: just check at the end
            pass
        
        def backtrack(idx):
            if time.time() - start_time > time_limit - 3:
                return False
            
            if idx == n:
                return True
            
            v = idx
            # Try each color
            for c in range(k):
                # Check if color c conflicts with already-colored neighbors
                valid = True
                for u in adj[v]:
                    if coloring[u] != -1 and coloring[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                coloring[v] = c
                color_to_vertices[c].add(v)
                
                # Quick check: for each other color c2, check if the subgraph
                # induced by {c, c2} has a cycle. We can do this by checking
                # if the number of edges in that subgraph exceeds the number
                # of vertices minus the number of connected components.
                
                # For efficiency, we'll do a lighter check: just verify that
                # we're not creating an obvious cycle. A full check is done at the end.
                
                if backtrack(idx + 1):
                    return True
                
                coloring[v] = -1
                color_to_vertices[c].discard(v)
            
            return False
        
        if backtrack(0):
            # Verify the coloring
            if _check_valid_coloring(coloring, n, edges):
                return coloring
    
    return None

def construct():
    n = P['n']
    edges = P['edges']
    
    # Try to find a valid coloring with few colors
    # Start with a reasonable upper bound
    
    # First, try a simple greedy approach with increasing k
    best_coloring = None
    
    # Try k from 2 upward
    for k in range(2, 11):
        coloring = _mip_search(n, edges, max_k=k, time_limit=55)
        if coloring is not None:
            best_coloring = coloring
            break
    
    if best_coloring is None:
        # Fallback: use all distinct colors (always valid)
        best_coloring = list(range(n))
    
    # Ensure we return Python ints
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
