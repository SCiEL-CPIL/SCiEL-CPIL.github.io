import time
import random
import math
import numpy as np

def _is_c4_free(N, n):
    """Check if graph with adjacency bitmasks N is C4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            common = N[i] & N[j]
            if common != 0 and (common & (common - 1)) != 0:
                return False
    return True

def _min_degree(deg):
    return min(deg)

def _edges_from_adj(N, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (N[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _can_add_edge(N, u, v):
    """Check if edge (u,v) can be added while maintaining C4-free property."""
    if (N[u] >> v) & 1:
        return False
    
    # |N(u) ∩ N(v)| must be ≤ 1
    common = N[u] & N[v]
    if common != 0 and (common & (common - 1)) != 0:
        return False
    
    # For all w in N(v), N(u) ∩ N(w) must be empty
    mask = N[v]
    while mask:
        lsb = mask & -mask
        w = lsb.bit_length() - 1
        if N[u] & N[w]:
            return False
        mask ^= lsb
    
    # For all w in N(u), N(v) ∩ N(w) must be empty
    mask = N[u]
    while mask:
        lsb = mask & -mask
        w = lsb.bit_length() - 1
        if N[v] & N[w]:
            return False
        mask ^= lsb
    
    return True

def _greedy_build_structured(target_n, min_deg_needed, seed_offset):
    """Build a C4-free graph using structured greedy approach."""
    random.seed(seed_offset)
    
    N = [0] * target_n
    deg = [0] * target_n
    
    def can_add(u, v):
        return _can_add_edge(N, u, v)
    
    def add_edge(u, v):
        N[u] |= (1 << v)
        N[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    
    # Greedy loop with degree balancing
    for _ in range(500):
        min_deg = min(deg)
        if min_deg >= min_deg_needed:
            break
        
        low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
        if not low_vertices:
            break
        
        u = random.choice(low_vertices)
        
        # Find valid candidates, prefer those that help balance degrees
        candidates = []
        for v in range(target_n):
            if v == u:
                continue
            if can_add(u, v):
                candidates.append((v, deg[v]))
        
        if not candidates:
            return None, None
        
        # Sort by degree (prefer lower degree vertices to balance)
        candidates.sort(key=lambda x: x[1])
        v = candidates[0][0]
        add_edge(u, v)
    
    if min(deg) >= min_deg_needed and _is_c4_free(N, target_n):
        return N, deg
    return None, None

def _local_search(N, deg, target_n, min_deg_needed, max_iter=2000):
    """Local search to improve minimum degree while maintaining C4-free property."""
    # Try to add edges to low-degree vertices
    for _ in range(max_iter):
        min_deg = min(deg)
        if min_deg >= min_deg_needed:
            break
        
        low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
        if not low_vertices:
            break
        
        u = random.choice(low_vertices)
        
        # Find all valid neighbors
        valid_neighbors = []
        for v in range(target_n):
            if v == u:
                continue
            if _can_add_edge(N, u, v):
                valid_neighbors.append(v)
        
        if not valid_neighbors:
            # Try removing an edge from u to free up space
            if deg[u] > 0:
                # Pick a random neighbor to remove
                mask = N[u]
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                N[u] &= ~(1 << w)
                N[w] &= ~(1 << u)
                deg[u] -= 1
                deg[w] -= 1
            continue
        
        v = random.choice(valid_neighbors)
        N[u] |= (1 << v)
        N[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    
    if min(deg) >= min_deg_needed and _is_c4_free(N, target_n):
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Number of restarts depends on time available
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Try structured greedy build
            N, deg = _greedy_build_structured(target_n, min_deg_needed, restart * 100 + target_n)
            
            if N is not None:
                # Verify and extract edges
                if _is_c4_free(N, target_n) and min(deg) >= min_deg_needed:
                    final_edges = _edges_from_adj(N, target_n)
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break
            
            # If greedy didn't work, try local search from a random start
            random.seed(restart * 200 + target_n + 50)
            N = [0] * target_n
            deg = [0] * target_n
            
            # Random initialization with some edges
            for _ in range(random.randint(50, 150)):
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u != v and _can_add_edge(N, u, v):
                    N[u] |= (1 << v)
                    N[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
            
            # Local search
            if _local_search(N, deg, target_n, min_deg_needed, max_iter=1000):
                final_edges = _edges_from_adj(N, target_n)
                
                if target_n > best_n:
                    best_n = target_n
                    best_edges = final_edges
                found = True
                break
        
        if best_n == 47:
            break
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}
