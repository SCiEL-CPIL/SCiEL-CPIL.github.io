import time
import random
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Build Edge List
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # 2. Build Conflict Graph
    # Two edges conflict if they are in the same 2D face.
    # We can determine this by checking if they share a vertex and the other two vertices form a face?
    # Or simpler: enumerate all faces and add conflicts.
    
    # Initialize neighbors as list of ints (bitsets)
    neighbors = [0] * n_edges
    
    # Enumerate faces
    # A face is defined by two dimensions i, j and a mask of the other d-2 bits.
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                # Add conflicts
                # e1 conflicts with e2, e3, e4
                # e2 conflicts with e1, e3, e4
                # etc.
                for a in [e1, e2, e3, e4]:
                    for b in [e1, e2, e3, e4]:
                        if a != b:
                            neighbors[a] |= (1 << b)
                            
    # 3. Branch and Bound for Maximum Independent Set
    # State: candidates (int), current_size (int)
    # Global: best_size, best_set (int)
    
    best_size = 0
    best_set = 0
    
    start_time = time.time()
    timeout = 115
    
    # Heuristic: Order vertices by degree?
    # For bitsets, we can just pick the lowest set bit, or the one with highest degree in the induced subgraph.
    # Computing degree in induced subgraph is expensive.
    # Let's use a static ordering or just pick the first available.
    # To improve, we can pre-sort vertices by degree and always pick the first available in that order.
    
    # Let's create a list of vertices sorted by degree descending.
    degrees = [bin(neighbors[i]).count('1') for i in range(n_edges)]
    order = sorted(range(n_edges), key=lambda x: -degrees[x])
    
    # To pick the "best" vertex from candidates, we need to find the first vertex in `order` that is in `candidates`.
    # This is O(N) per step. N=448. This is acceptable.
    
    def get_vertex(candidates):
        for v in order:
            if (candidates >> v) & 1:
                return v
        return -1

    def greedy_mis(candidates):
        # Greedily construct an independent set from candidates
        # This gives a lower bound, but we need an upper bound for pruning.
        # Actually, for MIS, a common upper bound is the number of vertices.
        # A better upper bound is the size of a maximum matching? No.
        # Let's use the simple bound: popcount(candidates).
        # If we want a tighter bound, we can use the fact that if we color the graph, 
        # the number of colors is not a bound for MIS.
        # However, we can use the "branch and reduce" technique.
        
        # For now, let's stick to popcount but use a good vertex ordering.
        return bin(candidates).count('1')

    def branch(candidates, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() - start_time > timeout:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            
        if candidates == 0:
            return
            
        # Prune
        if current_size + bin(candidates).count('1') <= best_size:
            return
            
        v = get_vertex(candidates)
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates without v and without neighbors of v
        new_candidates = candidates & ~(1 << v) & ~neighbors[v]
        branch(new_candidates, current_size + 1, current_set | (1 << v))
        
        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        branch(new_candidates, current_size, current_set)

    # Initial candidates: all edges
    all_candidates = (1 << n_edges) - 1
    
    branch(all_candidates, 0, 0)
    
    # If B&B didn't finish or didn't find a good solution, fallback to local search?
    # The B&B should find the optimal or near-optimal solution.
    # However, 448 nodes might be too big for pure Python B&B even with pruning.
    # Let's add a time check and if it times out, return the best found.
    # If best_size is 0, we need a fallback.
    
    if best_size == 0:
        # Fallback: Greedy Set Cover (from original) to get a valid solution
        # This ensures we return something valid.
        removed_mask = 0
        face_hit = [0] * (d * (d - 1) // 2 * (2 ** (d - 2))) # Approx count
        # Rebuild face list for this? No, just use the neighbor structure?
        # Actually, the original code had a good greedy init. Let's reuse the logic if needed.
        # But B&B should find at least something.
        
    # Construct result
    result = []
    for i in range(n_edges):
        if (best_set >> i) & 1:
            result.append(list(edges[i]))
            
    return result
