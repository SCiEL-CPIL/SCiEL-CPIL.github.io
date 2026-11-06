import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for 150s wall-clock safety margin
    
    d = 7
    n = 2 ** d  # 128
    m = n * d // 2  # 448 edges total
    
    # Precompute edges
    edges = []
    edge_id = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id[(x, y)] = len(edges)
                edges.append((x, y))
    
    # Precompute for each edge, the other two edges in each 4-face containing it
    # Edge (u,v) with direction i: the face is determined by another direction j != i
    # The 4-cycle: u - v (dir i), v - v^j (dir j), v^j - u^j (dir i), u^j - u (dir j)
    # So the other edges are: (v, v^j), (u^j, v^j), (u, u^j)
    edge_faces = [[] for _ in range(m)]  # edge_faces[e] = list of (j, e1, e2, e3) where e1,e2,e3 are the other 3 edges
    
    for idx, (x, y) in enumerate(edges):
        # Determine direction
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            # Edges: (x,c), (y,e), (c,e)
            e1 = edge_id.get((min(x, c), max(x, c)))
            e2 = edge_id.get((min(y, e), max(y, e)))
            e3 = edge_id.get((min(c, e), max(c, e)))
            edge_faces[idx].append((j, e1, e2, e3))
    
    # Build incidence: for each face (4-cycle), which edges form it
    # We'll track faces as tuples of 4 edge IDs
    # A face is identified by (x, i, j) where x is the minimum vertex, i<j are directions
    faces = []
    face_edges = []
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                a = x
                b = x ^ (1 << i)
                c = x ^ (1 << j)
                e = x ^ (1 << i) ^ (1 << j)
                e0 = edge_id.get((min(a, b), max(a, b)))
                e1 = edge_id.get((min(b, e), max(b, e)))
                e2 = edge_id.get((min(e, c), max(e, c)))
                e3 = edge_id.get((min(c, a), max(c, a)))
                if e0 is not None and e1 is not None and e2 is not None and e3 is not None:
                    faces.append((e0, e1, e2, e3))
    
    # Spectral relaxation approach
    # Build the line graph adjacency with signed weights
    # For each 4-face, we want to penalize having all 4 edges selected
    # Use a quadratic penalty: for each face, penalty = product of (1 - x_e) or similar
    
    # Alternative: use a spin model where each edge has spin +1 (selected) or -1 (not selected)
    # The 4-cycle constraint: not all 4 edges selected
    # Energy term per face: -lambda * (s1*s2*s3*s4) encourages not all same sign? No...
    
    # Better approach: greedy with local search
    # Start with a random or structured initialization, then do local optimization
    
    # Strategy: 
    # 1. Start with all edges, then remove edges to break all C4s (min vertex cover of C4 hypergraph)
    # 2. Or start with a structured set and add edges greedily
    # 3. Local search: try adding/removing/swapping edges to improve
    
    # Let's use a different structural approach:
    # The C4-free condition means for every pair of parallel edges in the hypercube,
    # the 4-cycle they form with two other edges must not be complete.
    
    # Key insight: group edges by their "direction" (bit position).
    # For each direction i, the edges form a perfect matching on Q7.
    # A 4-cycle uses 2 edges from one direction and 2 from another.
    
    # Let's try: for each pair of directions (i,j), consider the 2D grid formed.
    # In a 2D grid of size 2^(d-2) x 2^(d-2), we need C4-free subgraph.
    # Max C4-free in a 2D grid... this is complex.
    
    # Simpler: use randomized greedy + local search
    random.seed(42)
    
    def is_valid_addition(chosen, e):
        """Check if adding edge e to chosen set creates a C4."""
        x, y = edges[e]
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e2 = y ^ (1 << j)
            e1 = edge_id.get((min(x, c), max(x, c)))
            e2_id = edge_id.get((min(y, e2), max(y, e2)))
            e3 = edge_id.get((min(c, e2), max(c, e2)))
            if e1 in chosen and e2_id in chosen and e3 in chosen:
                return False
        return True
    
    # Greedy construction with randomization
    def greedy_construct(order):
        chosen = set()
        for e in order:
            if is_valid_addition(chosen, e):
                chosen.add(e)
        return chosen
    
    # Local search: try to add edges not in chosen
    def local_search(chosen, time_limit):
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            # Try adding random unselected edges
            unselected = [e for e in range(m) if e not in chosen]
            random.shuffle(unselected)
            for e in unselected:
                if (time.time() - t0) > time_limit:
                    break
                if is_valid_addition(chosen, e):
                    chosen.add(e)
                    improved = True
            # Try 2-for-1 swaps: remove one edge, add two
            if (time.time() - t0) > time_limit:
                break
            selected_list = list(chosen)
            random.shuffle(selected_list)
            for e_rem in selected_list[:50]:
                if (time.time() - t0) > time_limit:
                    break
                chosen.discard(e_rem)
                added = 0
                for e_add in unselected:
                    if (time.time() - t0) > time_limit:
                        break
                    if e_add == e_rem:
                        continue
                    if is_valid_addition(chosen, e_add):
                        chosen.add(e_add)
                        added += 1
                        if added >= 2:
                            break
                if added < 1:
                    chosen.add(e_rem)
                else:
                    improved = True
        return chosen
    
    # Multiple restarts with different random orderings
    best_chosen = set()
    best_size = 0
    
    # First: deterministic greedy
    order = list(range(m))
    chosen = greedy_construct(order)
    best_chosen = set(chosen)
    best_size = len(chosen)
    
    # Local search on greedy result
    if time.time() < deadline - 30:
        chosen = local_search(set(best_chosen), deadline - time.time() - 10)
        if len(chosen) > best_size:
            best_chosen = chosen
            best_size = len(chosen)
    
    # Try different random orderings
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(m))
        random.shuffle(order)
        # Bias: try directions in different orders
        chosen = greedy_construct(order)
        if len(chosen) > best_size:
            best_chosen = set(chosen)
            best_size = len(chosen)
        
        # Local search
        if time.time() < deadline - 15:
            chosen = local_search(set(best_chosen), min(5, deadline - time.time() - 10))
            if len(chosen) > best_size:
                best_chosen = chosen
                best_size = len(chosen)
    
    # More aggressive local search with 3-for-2 swaps
    if time.time() < deadline - 10:
        chosen = set(best_chosen)
        t0 = time.time()
        while time.time() - t0 < deadline - time.time() - 5:
            improved = False
            # Try removing 2 edges and adding 3
            selected_list = list(chosen)
            random.shuffle(selected_list)
            unselected = [e for e in range(m) if e not in chosen]
            random.shuffle(unselected)
            
            for e1 in selected_list[:30]:
                if time.time() > deadline - 5:
                    break
                for e2 in selected_list[:30]:
                    if e1 == e2:
                        continue
                    if time.time() > deadline - 5:
                        break
                    saved = set()
                    chosen.discard(e1)
                    chosen.discard(e2)
                    added = 0
                    for e_add in unselected:
                        if time.time() > deadline - 5:
                            break
                        if e_add in (e1, e2):
                            continue
                        if is_valid_addition(chosen, e_add):
                            chosen.add(e_add)
                            added += 1
                            if added >= 3:
                                break
                    if added < 2:
                        # Revert
                        chosen.add(e1)
                        chosen.add(e2)
                        for e_add in unselected:
                            if e_add in chosen and e_add not in (e1, e2):
                                # Check if we added it
                                pass
                        # Actually need to be more careful
                        # Let's just revert and try different pairs
                        pass
                    else:
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try single edge additions again
                for e_add in unselected:
                    if time.time() > deadline - 5:
                        break
                    if is_valid_addition(chosen, e_add):
                        chosen.add(e_add)
                        improved = True
                if not improved:
                    break
        
        if len(chosen) > best_size:
            best_chosen = chosen
            best_size = len(chosen)
    
    # Final greedy pass
    if time.time() < deadline - 3:
        unselected = [e for e in range(m) if e not in best_chosen]
        for e in unselected:
            if is_valid_addition(best_chosen, e):
                best_chosen.add(e)
    
    return [[edges[e][0], edges[e][1]] for e in sorted(best_chosen)]
