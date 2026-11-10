"""Rewritten program: simulated annealing with fast bicolored-forest check."""
import time
import random
import math

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Precompute adjacency
_ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    _ADJ[u].append(v)
    _ADJ[v].append(u)

# Precompute edge list as tuples for speed
_EDGE_LIST = [(u, v) for u, v in EDGES]


def _check_bicolored_forest(colors):
    """Check that the subgraph induced by any two colors is a forest.
    Returns True if valid, False otherwise.
    Uses union-find per color pair.
    """
    # For each pair of colors, collect edges between them and check for cycles
    # Group edges by color pair
    pair_edges = {}
    for u, v in _EDGE_LIST:
        cu, cv = colors[u], colors[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            if key not in pair_edges:
                pair_edges[key] = []
            pair_edges[key].append((u, v))
    
    for key, edges in pair_edges.items():
        # Union-find for these edges
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
    
    return True


def _is_valid(colors):
    """Full validation: proper coloring + no bicolored cycles."""
    for u, v in _EDGE_LIST:
        if colors[u] == colors[v]:
            return False
    return _check_bicolored_forest(colors)


def _cost(colors):
    """Cost function: number of same-color edges + number of bicolored cycles.
    Lower is better. Zero means valid.
    """
    # Count same-color edges
    same_count = 0
    for u, v in _EDGE_LIST:
        if colors[u] == colors[v]:
            same_count += 1
    
    if same_count > 0:
        return same_count * 10  # Heavy penalty for improper coloring
    
    # Count bicolored cycles (each cycle adds 1)
    pair_edges = {}
    for u, v in _EDGE_LIST:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        if key not in pair_edges:
            pair_edges[key] = []
        pair_edges[key].append((u, v))
    
    cycle_count = 0
    for key, edges in pair_edges.items():
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                cycle_count += 1
            else:
                parent[ru] = rv
    
    return cycle_count


def _count_distinct(colors):
    return len(set(colors))


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_colors = list(range(N))  # fallback: all distinct
    best_cost = _cost(best_colors)
    best_distinct = _count_distinct(best_colors)
    
    # Try k from 3 upward
    for k in [3, 4, 5, 6, 7]:
        if time.time() > deadline:
            break
        if k >= best_distinct:
            break
        
        # Multiple restarts with simulated annealing
        found_valid = False
        for restart in range(80):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Simulated annealing
            T = 5.0
            T_min = 0.01
            cooling = 0.995
            current_cost = _cost(colors)
            best_local = list(colors)
            best_local_cost = current_cost
            
            steps = 0
            max_steps = 30000
            while steps < max_steps and T > T_min and time.time() < deadline:
                # Pick a random vertex
                i = random.randint(0, N - 1)
                old_c = colors[i]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                
                colors[i] = new_c
                new_cost = _cost(colors)
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local = list(colors)
                else:
                    colors[i] = old_c
                
                T *= cooling
                steps += 1
            
            # Use best local
            if best_local_cost < best_cost:
                best_cost = best_local_cost
                best_colors = list(best_local)
                best_distinct = _count_distinct(best_local)
            
            if best_local_cost == 0:
                found_valid = True
                break
        
        if found_valid:
            break
    
    # Final validation
    if not _is_valid(best_colors):
        # Fallback: try a simple systematic approach
        # 4-coloring pattern for grid-like structure
        # The graph appears to be roughly a 5x10 grid
        # Let's try color = (i // 10) % 2 * 2 + i % 2
        colors = [(i // 10) % 2 * 2 + i % 2 for i in range(N)]
        if _is_valid(colors):
            best_colors = colors
        else:
            # Use all distinct as fallback
            best_colors = list(range(N))
    
    return [int(c) for c in best_colors]
