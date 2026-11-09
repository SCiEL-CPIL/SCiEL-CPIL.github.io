"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
# EVOLVE-BLOCK-START

import random
import time
from itertools import combinations

def _check_non_2_colorable(n, edges):
    """
    Fast check if a 5-uniform hypergraph is NOT 2-colorable.
    Returns True if non-2-colorable, False otherwise.
    Uses backtracking with unit propagation and bitmasks for speed.
    """
    m = len(edges)
    if m == 0:
        return False
        
    # Precompute edge incidence
    # For each vertex, list of edge indices
    v_to_e = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_to_e[v].append(i)
            
    # For each edge, store the vertices as a tuple for quick access
    edge_verts = [tuple(e) for e in edges]

    # State:
    # color: list of -1, 0, 1
    # edge_mask: list of int (bit 0 for color 0, bit 1 for color 1)
    # edge_rem: list of int (count of unassigned vertices)
    
    # We use a recursive DFS with global state modification (undo/redo) or copying.
    # Copying is simpler but slower. For small n (<=12), copying lists of size m (<= ~100) is fast enough.
    
    def dfs(color, edge_mask, edge_rem):
        # Check for contradiction
        for i in range(m):
            if edge_rem[i] == 0:
                if edge_mask[i] != 3:
                    return False
            elif edge_rem[i] == 1:
                if edge_mask[i] == 1:
                    # Must be 1
                    # Find the unassigned vertex in this edge
                    for v in edge_verts[i]:
                        if color[v] == -1:
                            # Force v to 1
                            # This is a propagation step. 
                            # For simplicity in this fast check, we might just branch on it.
                            # But unit propagation is key.
                            pass
                elif edge_mask[i] == 2:
                    # Must be 0
                    pass
        # For performance, let's do a simpler check first:
        # If any edge is monochromatic (mask 1 or 2) and rem==0 -> Fail
        # If any edge has rem==1 and mask==1 -> The last one MUST be 1.
        # If any edge has rem==1 and mask==2 -> The last one MUST be 0.
        
        # Let's implement a simple propagation loop inside DFS.
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if edge_rem[i] == 0:
                    if edge_mask[i] != 3:
                        return False
                elif edge_rem[i] == 1:
                    if edge_mask[i] == 1:
                        # Force unassigned to 1
                        for v in edge_verts[i]:
                            if color[v] == -1:
                                color[v] = 1
                                edge_rem[i] -= 1
                                edge_mask[i] |= 2
                                for j in v_to_e[v]:
                                    if j != i:
                                        edge_mask[j] |= 2
                                        edge_rem[j] -= 1
                                changed = True
                                break
                    elif edge_mask[i] == 2:
                        # Force unassigned to 0
                        for v in edge_verts[i]:
                            if color[v] == -1:
                                color[v] = 0
                                edge_rem[i] -= 1
                                edge_mask[i] |= 1
                                for j in v_to_e[v]:
                                    if j != i:
                                        edge_mask[j] |= 1
                                        edge_rem[j] -= 1
                                changed = True
                                break
                    # If mask is 3, it's satisfied, rem can be anything >0, but if rem becomes 0 later, it's fine.
                    # Wait, if mask is 3, the edge is satisfied regardless of remaining vertices.
                    # So we don't need to track rem for satisfied edges? 
                    # Yes, if mask==3, the edge is "done".
        
        # Check for contradictions again after propagation
        for i in range(m):
            if edge_rem[i] == 0 and edge_mask[i] != 3:
                return False

        # Find unassigned vertex
        unassigned = [v for v in range(n) if color[v] == -1]
        if not unassigned:
            # All assigned. Check if all edges are satisfied.
            for i in range(m):
                if edge_mask[i] != 3:
                    return False
            return True # Found a valid coloring

        # Heuristic: Choose vertex in the most constrained edge
        # Find edge with min rem > 0
        min_rem = n
        best_edge = -1
        for i in range(m):
            if edge_rem[i] > 0 and edge_rem[i] < min_rem:
                min_rem = edge_rem[i]
                best_edge = i
        
        if best_edge == -1:
            # All edges satisfied?
            return True

        # Pick a vertex from best_edge that is unassigned
        candidates = [v for v in edge_verts[best_edge] if color[v] == -1]
        # Randomize to avoid bias? Or pick first.
        v = candidates[0]
        
        # Try 0
        color_b = color[:]
        mask_b = edge_mask[:]
        rem_b = edge_rem[:]
        color_b[v] = 0
        for j in v_to_e[v]:
            mask_b[j] |= 1
            rem_b[j] -= 1
        if dfs(color_b, mask_b, rem_b):
            return True
            
        # Try 1
        color_b[v] = 1
        for j in v_to_e[v]:
            mask_b[j] |= 2
            rem_b[j] -= 1
        if dfs(color_b, mask_b, rem_b):
            return True
            
        return False

    initial_color = [-1] * n
    initial_mask = [0] * m
    initial_rem = [5] * m
    
    # If dfs returns True, it found a coloring, so it IS 2-colorable.
    # We want to know if it is NOT 2-colorable.
    is_colorable = dfs(initial_color, initial_mask, initial_rem)
    return not is_colorable

def construct():
    """
    Search for a small non-2-colorable 5-uniform hypergraph.
    """
    start_time = time.time()
    time_limit = 20.0 # seconds
    
    best_edges = None
    best_n = None
    
    # Try small n
    for n in range(10, 15):
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        current_edges = []
        # Start with a random subset
        # We want to find a small set.
        # Strategy: Add edges one by one. Check periodically.
        # To be efficient, we check every K edges.
        
        # Initialize with empty
        current_edges = []
        
        # We need to find a set that is non-2-colorable.
        # Random addition might take long to hit the threshold.
        # Better: Randomly select a subset of size K, check if non-2-colorable.
        # If yes, try to minimize by removing edges (greedy).
        
        # Let's try sizes from 29 upwards.
        for size in range(29, 60):
            if time.time() - start_time > time_limit:
                break
                
            # Try a few random samples of this size
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                    
                sample = random.sample(all_edges, size)
                # Convert to lists of ints
                sample_list = [list(e) for e in sample]
                
                # Check if non-2-colorable
                if _check_non_2_colorable(n, sample_list):
                    # We found a non-2-colorable hypergraph with 'size' edges.
                    # Try to minimize it by removing edges greedily.
                    current = sample_list
                    current_size = size
                    changed = True
                    while changed and time.time() - start_time < time_limit:
                        changed = False
                        # Try removing each edge
                        for i in range(len(current)):
                            candidate = current[:i] + current[i+1:]
                            if _check_non_2_colorable(n, candidate):
                                current = candidate
                                changed = True
                                break # Restart loop with new current
                            
                    if best_edges is None or len(current) < len(best_edges):
                        best_edges = current
                        best_n = n
                        # Update global best
                        # We can break if we find something very small, e.g., 29
                        if len(current) <= 29:
                            return {'n': best_n, 'edges': best_edges}
                            
    if best_edges:
        return {'n': best_n, 'edges': best_edges}
        
    # Fallback: Complete hypergraph on 9 vertices
    # 126 edges
    edges = [list(e) for e in combinations(range(9), 5)]
    return {'n': 9, 'edges': edges}

# EVOLVE-BLOCK-END
