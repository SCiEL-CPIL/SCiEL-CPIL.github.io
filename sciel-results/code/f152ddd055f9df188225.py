"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    
    # Precompute all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute for each edge, the other 3 edges of the 2D face it belongs to
    # For edge (u,v) differing in bit i, the face is spanned by bits i and j
    # The 4 edges are: (u,v), (u^bit_j, v^bit_j), (u^bit_j, u), (v^bit_j, v)
    # Wait, let me think more carefully.
    # If edge (u,v) differs in bit i, then for any other bit j != i:
    # The 2D face has vertices: u, v, u^(1<<j), v^(1<<j)
    # Edges: (u,v) [bit i], (u^(1<<j), v^(1<<j)) [bit i]
    #         (u, u^(1<<j)) [bit j], (v, v^(1<<j)) [bit j]
    
    face_edges = {}  # edge_idx -> list of 3 other edge indices
    for idx, (u, v) in enumerate(all_edges):
        # Determine which bit differs
        diff = u ^ v
        i = diff.bit_length() - 1
        others = []
        for j in range(d):
            if j == i:
                continue
            c = u ^ (1 << j)
            e = v ^ (1 << j)
            # Three other edges: (u,c), (v,e), (c,e)
            e1 = (min(u, c), max(u, c))
            e2 = (min(v, e), max(v, e))
            e3 = (min(c, e), max(c, e))
            others.append((edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3]))
        face_edges[idx] = others
    
    # Step 1: Deterministic algebraic construction
    # Use a linear map to select edges
    # For each vertex x, select edges in directions from a subspace
    # Key idea: if we pick a 3D subspace S of F_2^7, then for each x,
    # select edges in directions in S. This avoids C4s because...
    # Actually, let's use a different approach:
    # Partition the 7 dimensions into groups and use a structured selection
    
    # Better: use a greedy approach with randomization for initial construction
    random.seed(42)
    
    # Try multiple random orderings and keep the best
    best_count = 0
    best_selected = None
    
    num_tries = 50
    for trial in range(num_tries):
        if time.time() > deadline - 30:
            break
        
        # Random permutation of edges
        order = list(range(len(all_edges)))
        random.shuffle(order)
        
        selected = [False] * len(all_edges)
        count = 0
        
        for idx in order:
            u, v = all_edges[idx]
            # Check if adding this edge creates a C4
            conflict = False
            for o1, o2, o3 in face_edges[idx]:
                if selected[o1] and selected[o2] and selected[o3]:
                    conflict = True
                    break
            if not conflict:
                selected[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_selected = selected[:]
    
    # Start with best greedy result
    selected = best_selected if best_selected else [False] * len(all_edges)
    count = sum(selected)
    
    # Step 2: Local search improvement
    # Try to add edges that were rejected, or swap edges
    improved = True
    max_iters = 5000
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        if time.time() > deadline - 10:
            break
        
        # Try adding each unselected edge
        for idx in range(len(all_edges)):
            if selected[idx]:
                continue
            u, v = all_edges[idx]
            # Check conflicts
            conflicts = []
            for o1, o2, o3 in face_edges[idx]:
                if selected[o1] and selected[o2] and selected[o3]:
                    # This edge would create a C4
                    # To add it, we'd need to remove at least one of o1, o2, o3
                    # But that might not help net count
                    # For now, skip
                    conflicts.append((o1, o2, o3))
                    break
            
            if not conflicts:
                # Can add this edge directly
                selected[idx] = True
                count += 1
                improved = True
    
    # Step 3: Try swapping - remove one edge to allow adding multiple
    # For each edge, check if removing it allows adding 2+ edges
    for idx in range(len(all_edges)):
        if not selected[idx]:
            continue
        if time.time() > deadline - 10:
            break
        
        # Temporarily remove edge idx
        selected[idx] = False
        count -= 1
        
        # Try to add as many edges as possible
        added = []
        for jdx in range(len(all_edges)):
            if selected[jdx]:
                continue
            # Check if jdx can be added now
            can_add = True
            for o1, o2, o3 in face_edges[jdx]:
                if selected[o1] and selected[o2] and selected[o3]:
                    can_add = False
                    break
            if can_add:
                selected[jdx] = True
                count += 1
                added.append(jdx)
        
        # If we added more than 1 edge, keep the swap
        if len(added) > 1:
            # Keep the change (net gain)
            pass
        else:
            # Revert
            selected[idx] = True
            count += 1
            for jdx in added:
                selected[jdx] = False
                count -= 1
    
    # Build output
    out = []
    for idx in range(len(all_edges)):
        if selected[idx]:
            u, v = all_edges[idx]
            out.append([u, v])
    
    return out
# EVOLVE-BLOCK-END
