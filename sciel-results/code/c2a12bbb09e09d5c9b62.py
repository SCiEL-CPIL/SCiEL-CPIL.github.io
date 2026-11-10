import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    d = 7
    n = 1 << d  # 128 vertices
    
    # Precompute all edges: (u, v, dim) where u < v, v = u ^ (1<<dim)
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    # Total edges: 448
    # Precompute for each edge, which faces it belongs to
    # A face is identified by (a, b) where a, b differ in exactly 2 bits
    # The 4 edges of face (a,b) with midpoints c=a^bit1, e=a^bit2:
    # edges: (a,c), (c,b), (a,e), (e,b)
    
    # For fast lookup: for each edge (u,v,dim_i), the faces it's in are
    # for each j != i: face with corners u, v, u^j, v^j
    # Face key: sorted pair of the two "corner" vertices at distance 2
    
    # Better approach: for each edge (u,v,i), precompute the list of
    # (other_edge1, other_edge2, other_edge3) that would form a C4 with it
    # Actually, for C4 detection: adding edge (u,v,i) creates a C4 iff
    # exists j!=i such that (u, u^j), (v, v^j), (u^j, v^j) are all selected
    
    # Let's use a different representation for speed:
    # selected[u][i] = 1 if edge (u, u^(1<<i)) is selected (for u < u^(1<<i))
    # Actually, let's use: sel[u][i] where the edge connects u and u^(1<<i)
    # We can store it as a set of (min(u,v), i) or just a 2D structure
    
    # Fast approach: use bitsets
    # For each vertex u and dimension i, edge exists between u and u^(1<<i)
    # Let's use: adj[u] = bitmask of dimensions where edge is selected
    # But we need to check if edge (u,v) is selected where v = u^(1<<i)
    # So: is_selected(u, i) = (adj[u] >> i) & 1, but we need consistency
    
    # Let me use: sel = [0]*n where sel[u] is a bitmask of dimensions
    # Edge (u, u^(1<<i)) is selected iff (sel[u] >> i) & 1
    # But this is asymmetric. Let's ensure: for edge between u and v=u^(1<<i),
    # we store it in both sel[u] and sel[v] at bit i.
    
    # Actually simpler: use a set of frozenset or tuple keys
    # For speed, let's use: sel[u][i] for the edge from u in direction i
    # where we only store it once: if u < u^(1<<i), store in sel[u][i]
    # else store in sel[v][i] where v = u^(1<<i)
    
    # Let me just use a set of (min(u,v), i) for O(1) lookup
    # And for C4 check: adding edge (u,v,i) with u<v:
    #   for each j != i:
    #     c = u ^ (1<<j), e = v ^ (1<<j)
    #     check if (min(u,c), j), (min(v,e), j), (min(c,e), ?) are in set
    #     The edge (c,e): c and e differ in bit i, so edge is (min(c,e), i)
    
    # Precompute for each edge index, the list of (j, edge1_key, edge2_key, edge3_key)
    # where edge1, edge2, edge3 are the other 3 edges of the C4
    
    # Let's precompute edge keys
    def edge_key(u, v):
        if u < v:
            # find which bit differs
            diff = u ^ v
            i = diff.bit_length() - 1
            return (u, i)
        else:
            diff = u ^ v
            i = diff.bit_length() - 1
            return (v, i)
    
    # Precompute for each edge (u,v,i) with u<v, the C4 partners
    # edge (u,v,i): for each j != i:
    #   c = u ^ (1<<j), e = v ^ (1<<j)
    #   The 4 edges of the face: (u,c,j), (c,v,i), (u,e,j), (e,v,i)
    #   Wait, that's not right. The face has vertices u, v, c, e where
    #   u and v differ in bit i, c = u^(1<<j), e = v^(1<<j)
    #   Edges: (u,c) in dim j, (u,e) in dim j? No...
    #   u to c: differ in bit j -> edge (min(u,c), j)
    #   u to e: u and e = v^(1<<j) = u^(1<<i)^(1<<j), differ in bits i and j -> NOT an edge!
    
    # I made an error. Let me reconsider.
    # A 2D face in Q7 is defined by two dimensions i,j. The 4 vertices are:
    # a, a^(1<<i), a^(1<<j), a^(1<<i)^(1<<j)
    # The 4 edges are:
    # (a, a^(1<<i)) in dim i
    # (a, a^(1<<j)) in dim j  
    # (a^(1<<i), a^(1<<i)^(1<<j)) in dim j
    # (a^(1<<j), a^(1<<i)^(1<<j)) in dim i
    
    # So if I'm adding edge (u,v) where v = u^(1<<i), it participates in faces
    # for each j != i. The face is determined by a = u (or the appropriate corner).
    # If u < v, the face has corners: u, v=u^(1<<i), u^(1<<j), v^(1<<j)
    # The 4 edges:
    # e1 = (u, u^(1<<j)) -> dim j
    # e2 = (u, v) -> dim i  [this is the edge we're adding]
    # e3 = (u^(1<<j), v^(1<<j)) -> dim i
    # e4 = (v, v^(1<<j)) -> dim j
    
    # So to check if adding (u,v,i) completes a C4, for each j != i:
    #   check if e1, e3, e4 are all already selected
    
    # Precompute for each edge its C4 constraints
    edge_c4 = []  # for each edge, list of (j, key1, key3, key4)
    for (u, v, i) in all_edges:
        constraints = []
        for j in range(d):
            if j == i:
                continue
            c = u ^ (1 << j)
            e = v ^ (1 << j)
            # e1: (u, c) in dim j
            key1 = (min(u, c), j)
            # e3: (c, e) in dim i  [c and e differ in bit i]
            key3 = (min(c, e), i)
            # e4: (v, e) in dim j
            key4 = (min(v, e), j)
            constraints.append((key1, key3, key4))
        edge_c4.append(constraints)
    
    # Now do randomized greedy with multiple restarts
    best_count = 0
    best_edges = None
    rng = random.Random(42)
    
    def try_order(order):
        sel = set()
        count = 0
        for idx in order:
            u, v, i = all_edges[idx]
            # Check C4 constraints
            ok = True
            for (k1, k3, k4) in edge_c4[idx]:
                if k1 in sel and k3 in sel and k4 in sel:
                    ok = False
                    break
            if ok:
                sel.add((u, i))  # u < v guaranteed
                count += 1
        return count, sel
    
    # First try the natural order
    order = list(range(len(all_edges)))
    count, sel = try_order(order)
    best_count = count
    best_edges = set(sel)
    
    # Try multiple random orders
    n_edges = len(all_edges)
    for restart in range(2000):
        if time.time() > deadline:
            break
        order = list(range(n_edges))
        rng.shuffle(order)
        count, sel = try_order(order)
        if count > best_count:
            best_count = count
            best_edges = set(sel)
    
    # Local improvement: try to add more edges or swap
    # Actually, let's try a different approach: start from best and try to add edges
    # that were skipped, or do swaps
    
    # Try to build a better solution using a priority-based approach
    # Or try: for each random ordering, also try a "reverse" pass to add back
    
    # Let's try multiple more sophisticated strategies
    # Strategy: random ordering but with bias toward edges that are in fewer C4s
    # or random with local search
    
    # Local search: from best_edges, try removing one edge and adding two others
    if best_edges is not None:
        current_sel = set(best_edges)
        current_count = len(current_sel)
        
        # Build adjacency: for each edge, which other edges conflict with it
        # Two edges conflict if they share a C4 with two other edges
        # This is complex. Let's just try random removals and re-greedy
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Try: remove a random edge from current, then greedily add back
            if len(current_sel) == 0:
                break
            removed = rng.choice(list(current_sel))
            current_sel.remove(removed)
            
            # Now try to add edges in random order
            order = list(range(n_edges))
            rng.shuffle(order)
            new_sel = set(current_sel)
            for idx in order:
                u, v, i = all_edges[idx]
                key = (u, i)
                if key in new_sel:
                    continue
                ok = True
                for (k1, k3, k4) in edge_c4[idx]:
                    if k1 in new_sel and k3 in new_sel and k4 in new_sel:
                        ok = False
                        break
                if ok:
                    new_sel.add(key)
            
            if len(new_sel) > current_count:
                current_count = len(new_sel)
                current_sel = new_sel
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_sel)
            else:
                # Revert
                current_sel.add(removed)
    
    # Convert best_edges to output format
    result = []
    for (u, i) in best_edges:
        v = u ^ (1 << i)
        result.append([u, v])
    
    return result
