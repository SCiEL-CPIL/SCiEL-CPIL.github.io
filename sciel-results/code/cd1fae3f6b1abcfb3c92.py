import time
import math
import numpy as np
from collections import defaultdict

# Graph Definition
P = {
    'n': 50,
    'edges': [
        [0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3],
        [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47],
        [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14],
        [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12],
        [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21],
        [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19],
        [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28],
        [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26],
        [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35],
        [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33],
        [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42],
        [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40],
        [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49],
        [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]
    ]
}

N = P['n']
EDGES = np.array(P['edges'])
N_EDGES = len(EDGES)

# Precompute adjacency lists and sets for fast access
ADJ = [[] for _ in range(N)]
ADJ_SET = [set() for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)
    ADJ_SET[u].add(v)
    ADJ_SET[v].add(u)

# Precompute edge color pair indices? No, colors change.
# We need a fast way to check "bicolored cycles".
# Strategy: For a given coloring, group edges by color pair.
# A subgraph is a forest iff E <= V - C (where C is number of connected components).
# Or simpler: Union-Find is O(E alpha(V)). For 75 edges, this is very fast.
# But we want to do this inside a tight loop.

def _is_forest(edges_subset, colors):
    """
    Check if the subgraph induced by edges where colors match the pair is a forest.
    We will check this for all pairs present in the coloring.
    """
    if not edges_subset.size:
        return True
    
    # Group edges by color pair
    # To make this fast, we only check pairs that actually exist in the current coloring.
    # We can precompute the set of active color pairs.
    
    parent = list(range(N))
    rank = [0] * N
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Iterate only over edges that are part of a bicolored pair? 
    # Actually, ANY edge connects two colors. 
    # The condition is: For EVERY pair (c1, c2), the edges with those colors must form a forest.
    # This is equivalent to saying: The graph formed by edges of color pair (c1, c2) has no cycles.
    
    # Optimization: Instead of checking all pairs separately (which is redundant work on edges),
    # we can just check the whole graph? No, the constraint is per-pair.
    # If we have colors 0 and 1, edges between 0 and 1 must be acyclic.
    # If we have colors 0 and 2, edges between 0 and 2 must be acyclic.
    
    # We can iterate through all edges, and for each edge, if it's the one that creates a cycle 
    # in its specific color-pair subgraph, it's invalid.
    
    # Better approach:
    # For each color pair present, run Union Find on the subset of edges for that pair.
    # Since N=50 and E=75, this is very small.
    
    # Identify active pairs
    pairs_present = set()
    for u, v in edges_subset:
        cu = colors[u]
        cv = colors[v]
        if cu != cv:
            pairs_present.add((min(cu, cv), max(cu, cv)))
        else:
            # Adjacent vertices must have different colors. 
            # If cu == cv, it's an immediate violation.
            return False
            
    for c1, c2 in pairs_present:
        # Reset Union Find
        parent = list(range(N))
        rank = [0] * N
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        for u, v in edges_subset:
            cu = colors[u]
            cv = colors[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True

def _check_valid_numpy(colors):
    """
    Fast validity check.
    1. No adjacent same color.
    2. No bicolored cycles.
    """
    # 1. Adjacency check
    # Vectorized: colors[u] != colors[v] for all edges
    u, v = EDGES[:, 0], EDGES[:, 1]
    if np.any(colors[u] == colors[v]):
        return False
    
    # 2. Bicolored cycle check
    # We use the Python list based check for now as N is small.
    # Converting back to list for the union-find part is acceptable given N=50.
    # The bottleneck is usually the search space, not the check for N=50.
    return _is_forest(EDGES, colors)

def _initial_cayley_coloring():
    """
    Generate a valid coloring using the Cayley graph property.
    Graph is on Z_50. Edges are steps 1, 7, 43 (-7).
    c(v) = (a*v) % k.
    Valid if gcd(k, a) == 1 and gcd(k, 7) == 1.
    Smallest k is 6 (with a=1 or 5).
    """
    k = 6
    a = 1
    return [int((a * v) % k) for v in range(N)]

def _simulated_annealing(initial_colors, max_time):
    """
    Perform Simulated Annealing to minimize the number of distinct colors.
    """
    colors = np.array(initial_colors, dtype=np.int32)
    best_colors = colors.copy()
    best_count = len(set(colors.tolist()))
    
    start_time = time.time()
    T = 1.0
    T_min = 0.001
    alpha = 0.995
    
    # Count colors initially
    color_counts = [0] * 50
    for c in colors:
        color_counts[c] += 1
        
    current_count = best_count
    
    while time.time() - start_time < max_time and T > T_min:
        # Pick a random vertex
        v = np.random.randint(N)
        old_c = colors[v]
        
        # Pick a new color
        # We want to try reducing colors.
        # Strategy: Try to change v to a color that is already used, preferably one with high count, 
        # or a new low color if it reduces total distinct colors? 
        # Actually, minimizing distinct colors means we want to map vertices to existing colors if possible.
        # If we introduce a new color, we increase the count.
        
        # Let's try all existing colors first, then maybe a new one if it leads to a valid state 
        # and we are "cooling" (unlikely to help reduce count unless we swap later).
        # Better: Just try random colors from 0..49.
        
        new_c = np.random.randint(50)
        
        if new_c == old_c:
            T *= alpha
            continue
            
        # Apply move
        colors[v] = new_c
        
        # Check validity
        is_valid = _check_valid_numpy(colors)
        
        if is_valid:
            new_distinct = len(set(colors.tolist()))
            delta_count = new_distinct - current_count
            
            if delta_count <= 0:
                # Improvement or same
                # If improvement, update best
                if new_distinct < best_count:
                    best_count = new_distinct
                    best_colors = colors.copy()
                
                current_count = new_distinct
                # Accept always if valid and not worse? 
                # If delta > 0, we increased colors. We should only accept if it helps escape local minima?
                # Actually, if we increase colors, we move away from the goal. 
                # Standard SA accepts worse states with prob exp(-delta/T).
                # Here "cost" is number of colors. We want to minimize.
                # So delta = new_count - old_count.
                # If delta > 0, accept with prob exp(-delta/T).
            else:
                # Invalid state
                # Rollback
                colors[v] = old_c
                T *= alpha
                continue
        
        if not is_valid:
            colors[v] = old_c
            T *= alpha
            continue
            
        # Calculate acceptance probability for valid states that increased color count
        # (If delta_count < 0, we already accepted and updated best. We keep it as current state too.)
        if delta_count > 0:
            prob = math.exp(-delta_count / T)
            if np.random.random() < prob:
                # Accept worse state (more colors) to escape local opt
                current_count = new_distinct
            else:
                # Reject
                colors[v] = old_c
        # If delta_count <= 0, we already kept the new state in `colors`.
        
        T *= alpha
        
    return best_colors.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # 1. Generate initial valid coloring
    init_colors = _initial_cayley_coloring()
    
    # Verify it (just in case)
    if not _check_valid_numpy(np.array(init_colors)):
        # Fallback: Identity coloring is always valid (all distinct)
        return [int(i) for i in range(N)]
        
    # 2. Run Simulated Annealing to minimize colors
    # Allocate most of the time to SA
    sa_time = deadline - time.time() - 5.0
    if sa_time > 1.0:
        result = _simulated_annealing(init_colors, sa_time)
    else:
        result = init_colors
        
    # 3. Final Verification
    result_np = np.array(result, dtype=np.int32)
    if _check_valid_numpy(result_np):
        # Ensure types are Python ints
        return [int(x) for x in result]
    else:
        # Fallback to initial valid coloring
        return [int(x) for x in init_colors]
