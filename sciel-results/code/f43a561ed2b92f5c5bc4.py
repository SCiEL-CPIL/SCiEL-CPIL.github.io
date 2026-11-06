import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    d = 7
    n_vertices = 1 << d  # 128
    n_edges_total = n_vertices * d // 2  # 448
    
    # Precompute all edges
    edges = []
    edge_set = set()
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_set.add((u, v))
    
    # Precompute all C4 faces
    # A C4 is determined by two vertices differing in exactly 2 bits
    # The 4 edges of the C4 are the edges connecting these 4 vertices
    c4s = []  # Each C4 is a frozenset of 4 edges
    c4_by_edge = {}  # edge -> list of C4 indices
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v = u ^ (1 << i) ^ (1 << j)
                if u < v:
                    # The 4 vertices of this C4: u, u^(1<<i), u^(1<<j), v
                    a = u
                    b = u ^ (1 << i)
                    c = u ^ (1 << j)
                    e = v
                    # 4 edges: (a,b), (a,c), (b,e), (c,e)
                    e1 = (min(a, b), max(a, b))
                    e2 = (min(a, c), max(a, c))
                    e3 = (min(b, e), max(b, e))
                    e4 = (min(c, e), max(c, e))
                    c4 = frozenset([e1, e2, e3, e4])
                    c4s.append(c4)
    
    # Build edge -> C4 index mapping
    for idx, c4 in enumerate(c4s):
        for edge in c4:
            if edge not in c4_by_edge:
                c4_by_edge[edge] = []
            c4_by_edge[edge].append(idx)
    
    # Greedy set cover: pick edges to remove that cover the most C4s
    # Start with all edges present, remove edges greedily
    removed = set()
    active_c4s = set(range(len(c4s)))  # C4s not yet "hit"
    
    # Count how many active C4s each edge hits
    edge_hit_count = {}
    for edge in edges:
        count = 0
        for c4_idx in c4_by_edge.get(edge, []):
            if c4_idx in active_c4s:
                count += 1
        edge_hit_count[edge] = count
    
    # Greedy: repeatedly remove the edge that hits the most active C4s
    remaining_edges = list(edges)
    while active_c4s and remaining_edges:
        # Find edge with max hit count
        best_edge = None
        best_count = 0
        for edge in remaining_edges:
            if edge in removed:
                continue
            count = 0
            for c4_idx in c4_by_edge.get(edge, []):
                if c4_idx in active_c4s:
                    count += 1
            if count > best_count:
                best_count = count
                best_edge = edge
        
        if best_edge is None or best_count == 0:
            break
        
        removed.add(best_edge)
        remaining_edges.remove(best_edge)
        
        # Update active C4s: remove C4s that now have an edge removed
        for c4_idx in c4_by_edge.get(best_edge, []):
            if c4_idx in active_c4s:
                active_c4s.discard(c4_idx)
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
    
    # Now we have a set of removed edges. The remaining edges should be C4-free.
    # Let's verify and build the result
    kept_edges = []
    for edge in edges:
        if edge not in removed:
            kept_edges.append(list(edge))
    
    # Verify no C4 is fully present
    kept_set = set(kept_edges)
    for c4 in c4s:
        if all(e in kept_set for e in c4):
            # This shouldn't happen, but if it does, remove one edge
            for e in c4:
                if e in kept_set:
                    kept_set.discard(e)
                    break
    
    kept_edges = [list(e) for e in kept_set]
    
    # Try to add back edges that don't create C4s
    if time.time() - start_time < time_limit - 10:
        # Try adding back removed edges in random order
        random.seed(42)
        removed_list = list(removed)
        random.shuffle(removed_list)
        
        for edge in removed_list:
            if time.time() - start_time > time_limit - 5:
                break
            # Check if adding this edge creates any C4
            can_add = True
            for c4_idx in c4_by_edge.get(edge, []):
                c4 = c4s[c4_idx]
                if all(e in kept_set for e in c4 if e != edge):
                    can_add = False
                    break
            if can_add:
                kept_set.add(edge)
    
    result = [list(e) for e in kept_set]
    return result
