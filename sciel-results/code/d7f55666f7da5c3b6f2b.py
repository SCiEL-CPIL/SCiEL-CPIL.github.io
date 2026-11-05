import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d
    num_edges = d * (2 ** (d - 1))  # 448
    
    # Generate all edges
    edges = []
    edge_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_map[(x, y)] = idx
    
    # Generate all 2-dimensional faces (C4s)
    # A face is defined by two distinct bit positions i, j and a base vertex v
    # where the bits i and j of v are 0.
    # The 4 vertices are:
    # v
    # v ^ (1 << i)
    # v ^ (1 << j)
    # v ^ (1 << i) ^ (1 << j)
    
    faces = []
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all vertices v such that bit i and bit j are 0
            # We can iterate over all vertices and filter, or construct directly.
            # Direct construction:
            # The other d-2 bits can be anything.
            mask_i = 1 << i
            mask_j = 1 << j
            
            # Iterate over all 2^(d-2) combinations of the other bits
            # To do this efficiently, we can iterate 0..2^d-1 and check if (v & mask_i) == 0 and (v & mask_j) == 0
            for v in range(n):
                if (v & mask_i) == 0 and (v & mask_j) == 0:
                    v1 = v
                    v2 = v ^ mask_i
                    v3 = v ^ mask_j
                    v4 = v ^ mask_i ^ mask_j
                    
                    # Edges:
                    # (v1, v2), (v3, v4) are parallel to i
                    # (v1, v3), (v2, v4) are parallel to j
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v3, v4), max(v3, v4))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v4), max(v2, v4))
                    
                    idxs = [edge_map[e1], edge_map[e2], edge_map[e3], edge_map[e4]]
                    face_idx = len(faces)
                    faces.append(idxs)
                    
                    for idx in idxs:
                        edge_to_faces[idx].append(face_idx)
    
    # Helper function to check if adding an edge is valid
    # selected: list of bools, size num_edges
    def can_add(edge_idx, selected):
        for f_idx in edge_to_faces[edge_idx]:
            f_edges = faces[f_idx]
            # Check if the other 3 edges are selected
            # f_edges contains 4 indices. One is edge_idx.
            # We can just check all 4, but we know edge_idx is not selected yet (we are checking if we CAN add it).
            # Actually, we are checking if adding it creates a C4.
            # So we check if the OTHER 3 are in selected.
            # It's faster to check all 4 and see if count is 3 (since edge_idx is 0).
            # Or just check the specific 3.
            
            # Optimization: Unroll or use a small loop
            cnt = 0
            for e_idx in f_edges:
                if e_idx != edge_idx and selected[e_idx]:
                    cnt += 1
                    if cnt == 3:
                        return False
            # If cnt < 3, it's safe
        return True

    # Helper to check if a swap is valid
    def can_swap(edge_out_idx, edge_in_idx, selected):
        # Temporarily remove edge_out
        selected[edge_out_idx] = False
        # Check if edge_in can be added
        res = can_add(edge_in_idx, selected)
        # Restore
        selected[edge_out_idx] = True
        return res

    best_edges = []
    best_count = 0
    
    # Phase 1: Random Greedy Restarts
    # We run multiple restarts to find a good local optimum
    # Then we try to improve with swaps
    
    num_restarts = 0
    while time.time() < deadline - 10:  # Leave 10s for final cleanup/return
        num_restarts += 1
        random.seed(num_restarts * 12345)
        
        # Shuffle edge indices
        edge_indices = list(range(num_edges))
        random.shuffle(edge_indices)
        
        selected = [False] * num_edges
        count = 0
        
        # Greedy fill
        for idx in edge_indices:
            if can_add(idx, selected):
                selected[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_edges = [i for i in range(num_edges) if selected[i]]
            
        # Phase 2: Local Search on the current solution
        # Try to add more edges
        # Then try swaps to unlock adds
        
        # We operate on the 'selected' array from the greedy phase
        # Note: best_edges is a copy. We should probably continue optimizing the current 'selected'
        # and update best if we find something better.
        
        improved = True
        swap_count = 0
        max_swaps_per_restart = 5000  # Limit to avoid getting stuck in one restart
        
        while improved and time.time() < deadline - 5 and swap_count < max_swaps_per_restart:
            improved = False
            
            # 1. Try to add any edge
            # We can iterate through non-selected edges
            # To speed up, we can maintain a list of "blocked" edges?
            # For now, just scan.
            for idx in range(num_edges):
                if not selected[idx]:
                    if can_add(idx, selected):
                        selected[idx] = True
                        count += 1
                        improved = True
                        # After an add, we might be able to add more, so break and restart scan
                        break
            
            if improved:
                if count > best_count:
                    best_count = count
                    best_edges = [i for i in range(num_edges) if selected[i]]
                continue
            
            # 2. No add possible. Try a swap.
            # Pick a random selected edge to remove
            # Pick a random non-selected edge to add
            # This is a random walk.
            
            # To be more effective, we can try to swap an edge that is "critical" for many blocked edges
            # But random is simple and often works.
            
            # Get list of selected and non-selected
            # To avoid O(N) scan every time, we can keep lists, but they change.
            # Given N=448, scanning is fine.
            
            sel_list = [i for i in range(num_edges) if selected[i]]
            non_sel_list = [i for i in range(num_edges) if not selected[i]]
            
            if not sel_list or not non_sel_list:
                break
                
            e_out = random.choice(sel_list)
            e_in = random.choice(non_sel_list)
            
            if can_swap(e_out, e_in, selected):
                selected[e_out] = False
                selected[e_in] = True
                swap_count += 1
                # The count remains the same, but we might be able to add more now
                # We don't update best_count here as count is same
                # But we set improved = True to continue the loop and try adds
                improved = True
        
    # Ensure we return a list of [u, v] pairs
    result = []
    for idx in best_edges:
        u, v = edges[idx]
        result.append([u, v])
        
    return result

# EVOLVE-BLOCK-END
