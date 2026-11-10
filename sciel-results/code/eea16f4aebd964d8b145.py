import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    
    # Generate all edges and assign each an index
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = len(edges)
                edges.append((u, v))
    
    n_edges = len(edges)
    
    # Precompute for each edge, the three other edges that would complete a C4
    # For edge (u,v) differing in bit i, and another bit j:
    #   w1 = u ^ (1<<j), w2 = v ^ (1<<j)
    #   The three companion edges are: (u,w1), (v,w2), (w1,w2)
    edge_companions = []  # edge_companions[e_idx] = list of (j_idx, k_idx, l_idx) for each j
    for e_idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        companions = []
        for j in range(d):
            if j == i:
                continue
            mask = 1 << j
            w1 = u ^ mask
            w2 = v ^ mask
            e1 = (min(u, w1), max(u, w1))
            e2 = (min(v, w2), max(v, w2))
            e3 = (min(w1, w2), max(w1, w2))
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            companions.append((idx1, idx2, idx3))
        edge_companions.append(companions)
    
    def can_add_fast(e_idx, selected):
        """Check if edge e_idx can be added without creating a C4."""
        for (a, b, c) in edge_companions[e_idx]:
            if selected[a] and selected[b] and selected[c]:
                return False
        return True
    
    def greedy_random():
        """One randomized greedy pass."""
        order = list(range(n_edges))
        random.shuffle(order)
        selected = [False] * n_edges
        count = 0
        for e_idx in order:
            if can_add_fast(e_idx, selected):
                selected[e_idx] = True
                count += 1
        return selected, count
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_selected = None
    best_count = 0
    
    # Phase 1: Many randomized greedy passes
    while time.time() < deadline - 10.0:
        selected, count = greedy_random()
        if count > best_count:
            best_count = count
            best_selected = selected
    
    # Phase 2: Local search - remove one edge, try to add others
    if best_selected is not None:
        for iteration in range(50):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            # Try removing each edge and see if we can add back more
            for e_remove in range(n_edges):
                if time.time() > deadline - 2.0:
                    break
                if not best_selected[e_remove]:
                    continue
                
                # Temporarily remove
                best_selected[e_remove] = False
                current_count = best_count - 1
                
                # Try to add edges that were previously blocked
                for e_add in range(n_edges):
                    if best_selected[e_add]:
                        continue
                    if can_add_fast(e_add, best_selected):
                        best_selected[e_add] = True
                        current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    improved = True
                    break
                else:
                    # Restore
                    best_selected[e_remove] = True
            
            if not improved:
                # Try a few more random restarts if no improvement
                for _ in range(3):
                    if time.time() > deadline - 2.0:
                        break
                    selected, count = greedy_random()
                    if count > best_count:
                        best_count = count
                        best_selected = selected
                        improved = True
                    if not improved:
                        break
    
    # Final validation and repair to ensure no C4s
    if best_selected is not None:
        # Check all 2D faces
        for u in range(n_vertices):
            for i in range(d):
                for j in range(i + 1, d):
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(u, v1), max(u, v1))
                    e2 = (min(u, v2), max(u, v2))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v3), max(v2, v3))
                    
                    idx1 = edge_index[e1]
                    idx2 = edge_index[e2]
                    idx3 = edge_index[e3]
                    idx4 = edge_index[e4]
                    
                    if best_selected[idx1] and best_selected[idx2] and best_selected[idx3] and best_selected[idx4]:
                        # Remove one edge to break the cycle
                        best_selected[idx4] = False
        
    # Convert to list of lists
    result = []
    for e_idx, (u, v) in enumerate(edges):
        if best_selected is not None and best_selected[e_idx]:
            result.append([u, v])
    
    return result
