import time
import random
from itertools import product

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115

def build_groetzsch_adj():
    """
    Returns adjacency list for the Grötzsch graph (11 vertices).
    """
    n = 11
    adj = [0] * n
    def add_edge(u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    # C5
    add_edge(0, 1)
    add_edge(1, 2)
    add_edge(2, 3)
    add_edge(3, 4)
    add_edge(4, 0)
    
    # Mycielski copies
    # 5 (copy of 0) connects to N(0) = {1, 4}
    add_edge(5, 1)
    add_edge(5, 4)
    
    # 6 (copy of 1) connects to N(1) = {0, 2}
    add_edge(6, 0)
    add_edge(6, 2)
    
    # 7 (copy of 2) connects to N(2) = {1, 3}
    add_edge(7, 1)
    add_edge(7, 3)
    
    # 8 (copy of 3) connects to N(3) = {2, 4}
    add_edge(8, 2)
    add_edge(8, 4)
    
    # 9 (copy of 4) connects to N(4) = {3, 0}
    add_edge(9, 3)
    add_edge(9, 0)
    
    # 10 connects to 5, 6, 7, 8, 9
    for i in range(5, 10):
        add_edge(10, i)
        
    return adj

def is_not_3_colorable_bitmask(adj, n=16):
    """
    Exact backtracking check for 3-colorability.
    Returns True if NOT 3-colorable.
    """
    colors = [-1] * n
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        
        # Heuristic: Pick vertex with most colored neighbors (most constrained)
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        
        return True
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask)

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    
    n_vertices = 16
    base_adj = build_groetzsch_adj()
    n_base = 11
    
    # Precompute neighborhood intersections for base graph
    can_saturate = [[False] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if not (base_adj[i] & (1 << j)):
                if (base_adj[i] & base_adj[j]) == 0:
                    can_saturate[i][j] = True
                    can_saturate[j][i] = True

    best_edges_count = 0
    best_composition = None
    
    # Generate compositions of 16 into 11 parts (k_i >= 1)
    def gen_compositions(remaining, num_parts):
        if num_parts == 1:
            yield [remaining]
            return
        for first in range(1, remaining - num_parts + 2):
            for rest in gen_compositions(remaining - first, num_parts - 1):
                yield [first] + rest

    for comp in gen_compositions(n_vertices, n_base):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Calculate edges
        edge_count = 0
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if (base_adj[i] & (1 << j)) or can_saturate[i][j]:
                    edge_count += comp[i] * comp[j]
                    
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_composition = comp

    if best_composition is None:
        return []

    # Construct the initial graph from the best composition
    edges = []
    bag_map = [0] * n_vertices
    current_idx = 0
    for i in range(n_base):
        for _ in range(best_composition[i]):
            bag_map[current_idx] = i
            current_idx += 1
            
    adj = [0] * n_vertices
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            bi = bag_map[i]
            bj = bag_map[j]
            if bi == bj:
                continue
            if (base_adj[bi] & (1 << bj)) or can_saturate[bi][bj]:
                edges.append([i, j])
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                
    # Local Search Phase: Try to add more edges
    # We iterate through potential edges that maintain triangle-freeness.
    # If adding an edge keeps the graph non-3-colorable, we accept it.
    
    improved = True
    while improved and time.time() - START_TIME < TIME_LIMIT - 5:
        improved = False
        # Find all candidate edges (u, v) such that N(u) and N(v) are disjoint
        candidates = []
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Try adding edge (u, v)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
            # Check if graph is still not 3-colorable
            if is_not_3_colorable_bitmask(adj, n_vertices):
                edges.append([u, v])
                improved = True
            else:
                # Revert
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                
    return edges
