"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_bicolor_cycle(coloring, n, adj, color_a, color_b):
    """Check if the subgraph induced by vertices with color_a or color_b has a cycle using union-find."""
    vertices = [v for v in range(n) if coloring[v] == color_a or coloring[v] == color_b]
    if not vertices:
        return False
    vset = set(vertices)
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return True  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False
    
    for u in vertices:
        for v in adj[u]:
            if v in vset and u < v:
                if union(u, v):
                    return True
    return False

def _is_valid_assignment(coloring, n, adj, k):
    """Check all constraints: adjacent differ, and no bicolored cycle for any color pair."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = list(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            if _has_bicolor_cycle(coloring, n, adj, colors_used[i], colors_used[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Fallback: Unique colors is always valid
    fallback = [int(i) for i in range(n)]
    
    # Use Local Search (Simulated Annealing) to minimize bicolored cycles
    # Cost function: Sum over all color pairs of (Edges - Vertices + Components)
    # which is the number of cycles in the induced subgraph.
    
    def _calc_cost(coloring, k):
        """Calculate the total number of bicolored cycles."""
        total_cycles = 0
        # Only check pairs of colors that are actually used
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                # Check subgraph induced by c1 and c2
                # Use Union-Find to count cycles
                parent = {}
                rank = {}
                
                def find(x):
                    if x not in parent:
                        parent[x] = x
                        rank[x] = 0
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx = find(x)
                    ry = find(y)
                    if rx == ry:
                        return True # Cycle
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return False
                
                cycles = 0
                for u in range(n):
                    if coloring[u] == c1 or coloring[u] == c2:
                        for v in adj[u]:
                            if u < v and (coloring[v] == c1 or coloring[v] == c2):
                                if union(u, v):
                                    cycles += 1
                total_cycles += cycles
        return total_cycles

    # Initialize with a random 4-coloring
    import random
    random.seed(42)
    k_initial = 4
    coloring = [random.randint(0, k_initial - 1) for _ in range(n)]
    
    # Ensure proper coloring initially? Not strictly necessary for SA, but helps.
    # We will allow invalid proper colorings but penalize them heavily?
    # The task says "Adjacent vertices must differ".
    # So we must maintain proper coloring.
    
    # Check if initial is proper
    def _is_proper(coloring):
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                return False
        return True
    
    # Repair initial coloring if needed
    if not _is_proper(coloring):
        for i in range(n):
            forbidden = set(coloring[nb] for nb in adj[i])
            coloring[i] = 0
            while coloring[i] in forbidden:
                coloring[i] += 1
    
    current_cost = _calc_cost(coloring, k_initial)
    best_coloring = coloring[:]
    best_cost = current_cost
    
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a random vertex
        v = random.randint(0, n - 1)
        
        # Pick a new color (different from neighbors)
        forbidden = set(coloring[nb] for nb in adj[v])
        # Try a random color from 0..49
        new_c = random.randint(0, 49)
        attempts = 0
        while new_c in forbidden and attempts < 10:
            new_c = random.randint(0, 49)
            attempts += 1
        if new_c in forbidden:
            continue
            
        old_c = coloring[v]
        if old_c == new_c:
            continue
            
        coloring[v] = new_c
        
        new_cost = _calc_cost(coloring, 50) # k doesn't matter for cost calc if we use sets
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = coloring[:]
                if best_cost == 0:
                    break
        else:
            coloring[v] = old_c
            
        temperature *= cooling
        if temperature < 0.01:
            break

    # Verify best solution
    if _is_valid_assignment(best_coloring, n, adj, 50):
        return [int(x) for x in best_coloring]
    
    # If local search failed to find a valid one, return fallback
    return fallback

# EVOLVE-BLOCK-END
