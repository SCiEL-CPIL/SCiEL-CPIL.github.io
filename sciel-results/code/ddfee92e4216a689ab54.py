import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    # All possible edges
    all_edges = list(combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_possible_edges = len(all_edges)
    
    # Precompute edge endpoints
    edge_u = np.array([e[0] for e in all_edges])
    edge_v = np.array([e[1] for e in all_edges])
    
    # Adjacency bitmask for each vertex (as integer bits)
    # adj_mask[i] has bit j set if edge (i,j) exists
    
    def build_adj_mask(edge_set):
        """Build adjacency bitmasks from a set of edge indices."""
        masks = [0] * n
        for idx in edge_set:
            u = all_edges[idx][0]
            v = all_edges[idx][1]
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    def max_triangle_sum(masks):
        """Compute max degree sum over all triangles using bitmasks.
        Returns (max_sum, worst_triangle_vertices)"""
        deg = [bin(m).count('1') for m in masks]
        max_s = 0
        worst_tri = None
        for u in range(n):
            for v in range(u+1, n):
                if not (masks[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = masks[u] & masks[v]
                # Only consider w > v to avoid duplicates
                common &= ~((1 << (v+1)) - 1)
                while common:
                    lsb = common & (-common)
                    w = lsb.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_s:
                        max_s = s
                        worst_tri = (u, v, w)
                    common ^= lsb
        return max_s, worst_tri

    # Initial configuration: K_{8,8} + one internal edge
    # K_{8,8} edges: all (u,v) where u in [0,8), v in [8,16)
    k88_edges = set()
    for u in range(8):
        for v in range(8, 16):
            k88_edges.add(edge_to_idx[(u, v)])
    
    # Add internal edge (0,1)
    init_edge_set = set(k88_edges)
    init_edge_set.add(edge_to_idx[(0, 1)])
    
    assert len(init_edge_set) == target_edges
    
    init_masks = build_adj_mask(init_edge_set)
    init_score, _ = max_triangle_sum(init_masks)
    
    best_score = init_score
    best_edge_set = set(init_edge_set)
    
    current_edge_set = set(init_edge_set)
    current_masks = build_adj_mask(current_edge_set)
    current_score = init_score
    
    rng = np.random.default_rng(42)
    
    # Precompute list of all edges for random selection
    all_edge_indices = list(range(num_possible_edges))
    
    # Targeted Local Search
    # We will alternate between:
    # 1. Targeted moves: Focus on the worst triangle.
    # 2. Random moves: To escape local minima.
    
    max_iterations = 200000
    iter_count = 0
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Determine if we should use targeted or random move
        # Use targeted moves 70% of the time
        if rng.random() < 0.7:
            # Identify worst triangle
            _, worst_tri = max_triangle_sum(current_masks)
            if worst_tri is None:
                # No triangles? Unlikely with 65 edges on 16 nodes, but handle it.
                # Fall back to random
                pass
            else:
                u, v, w = worst_tri
                # Generate targeted moves
                # Strategy A: Break the triangle by removing one of its edges
                # Strategy B: Swap an edge incident to a high-degree vertex in the triangle
                # with a non-incident edge.
                
                # Let's try to reduce the degree of the highest degree vertex in the triangle
                deg_u = bin(current_masks[u]).count('1')
                deg_v = bin(current_masks[v]).count('1')
                deg_w = bin(current_masks[w]).count('1')
                
                # Find vertex with max degree in triangle
                if deg_u >= deg_v and deg_u >= deg_w:
                    target_vertex = u
                elif deg_v >= deg_u and deg_v >= deg_w:
                    target_vertex = v
                else:
                    target_vertex = w
                
                # Find edges incident to target_vertex
                incident_edges = [e_idx for e_idx in current_edge_set 
                                  if all_edges[e_idx][0] == target_vertex or all_edges[e_idx][1] == target_vertex]
                
                if not incident_edges:
                    continue # Should not happen
                
                # Pick a random edge to remove from incident edges
                remove_idx = incident_edges[rng.integers(0, len(incident_edges))]
                
                # Pick an edge to add that is NOT incident to target_vertex
                # This helps reduce the degree of target_vertex
                valid_adds = [e_idx for e_idx in range(num_possible_edges) 
                              if e_idx not in current_edge_set 
                              and all_edges[e_idx][0] != target_vertex 
                              and all_edges[e_idx][1] != target_vertex]
                
                if not valid_adds:
                    continue
                
                add_idx = valid_adds[rng.integers(0, len(valid_adds))]
                
                # Apply move
                new_edge_set = set(current_edge_set)
                new_edge_set.discard(remove_idx)
                new_edge_set.add(add_idx)
                
                if len(new_edge_set) != target_edges:
                    continue
                
                new_masks = build_adj_mask(new_edge_set)
                new_score, _ = max_triangle_sum(new_masks)
                
                # Accept if better or equal (with some randomness to escape plateaus)
                if new_score < current_score or (new_score == current_score and rng.random() < 0.5):
                    current_edge_set = new_edge_set
                    current_masks = new_masks
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_edge_set = set(current_edge_set)
        
        else:
            # Random move
            remove_idx = rng.integers(0, num_possible_edges)
            add_idx = rng.integers(0, num_possible_edges)
            
            while add_idx == remove_idx:
                add_idx = rng.integers(0, num_possible_edges)
            
            if remove_idx not in current_edge_set:
                continue
            if add_idx in current_edge_set:
                continue
            
            new_edge_set = set(current_edge_set)
            new_edge_set.discard(remove_idx)
            new_edge_set.add(add_idx)
            
            if len(new_edge_set) != target_edges:
                continue
            
            new_masks = build_adj_mask(new_edge_set)
            new_score, _ = max_triangle_sum(new_masks)
            
            # Simple hill climbing for random moves to keep it tight
            if new_score < current_score:
                current_edge_set = new_edge_set
                current_masks = new_masks
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_edge_set = set(current_edge_set)
        
        iter_count += 1
    
    # Final polish: Try all single swaps for the best solution found
    # This is expensive, so we limit it to the last few seconds
    if time.time() < deadline - 2:
        current_edge_set = set(best_edge_set)
        current_masks = build_adj_mask(current_edge_set)
        current_score = best_score
        
        # Sort edges to ensure deterministic order for polish if needed
        edges_list = list(current_edge_set)
        
        # Limit polish iterations to avoid timeout
        polish_steps = 0
        while time.time() < deadline - 1 and polish_steps < 50:
            improved = False
            # Try removing each edge and adding the best possible edge
            for remove_idx in edges_list:
                if time.time() > deadline - 1:
                    break
                
                best_add_score = current_score
                best_add_idx = -1
                
                # Try adding each non-present edge
                # Optimization: Only try a subset of edges if time is tight?
                # For now, try all, but break early if we find a significant improvement?
                # No, we need the exact min.
                
                for add_idx in range(num_possible_edges):
                    if add_idx in current_edge_set:
                        continue
                    
                    new_edge_set = set(current_edge_set)
                    new_edge_set.discard(remove_idx)
                    new_edge_set.add(add_idx)
                    
                    if len(new_edge_set) != target_edges:
                        continue
                    
                    new_masks = build_adj_mask(new_edge_set)
                    new_score, _ = max_triangle_sum(new_masks)
                    
                    if new_score < best_add_score:
                        best_add_score = new_score
                        best_add_idx = add_idx
                
                if best_add_idx != -1:
                    # Apply the best swap
                    current_edge_set = set(current_edge_set)
                    current_edge_set.discard(remove_idx)
                    current_edge_set.add(best_add_idx)
                    current_masks = build_adj_mask(current_edge_set)
                    current_score = best_add_score
                    improved = True
                    break
            
            if not improved:
                break
            
            if current_score < best_score:
                best_score = current_score
                best_edge_set = set(current_edge_set)
            
            polish_steps += 1
    
    edges_out = [[all_edges[e][0], all_edges[e][1]] for e in best_edge_set]
    return {'n': n, 'edges': edges_out}
