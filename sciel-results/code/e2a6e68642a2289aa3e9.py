import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    # Map vertices to bit positions
    # We represent the set of edges as a list of bitmasks
    edge_masks = [0] * len(edges)
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
        
    # Precompute for each vertex, the list of edges it belongs to
    vertex_edges = [[] for _ in range(n)]
    for i, mask in enumerate(edge_masks):
        for v in range(n):
            if mask & (1 << v):
                vertex_edges[v].append(i)

    # State:
    # color[v] = -1 (unassigned), 0 (red), 1 (blue)
    # We use a list of lists for constraints? 
    # Simpler: Backtrack on vertices.
    # To speed up, we can use a simple recursive search with pruning.
    
    color = [-1] * n
    # We will assign colors to vertices 0..n-1
    # Optimization: If an edge has 4 vertices of one color, the 5th MUST be the other color (Unit Propagation)
    
    def propagate(color):
        # Repeat until no change
        changed = True
        while changed:
            changed = False
            for e_idx, mask in enumerate(edge_masks):
                # Count colors in this edge
                reds = 0
                blues = 0
                uncolored = []
                for v in range(n):
                    if mask & (1 << v):
                        if color[v] == 0:
                            reds += 1
                        elif color[v] == 1:
                            blues += 1
                        else:
                            uncolored.append(v)
                
                if reds == 5: return False # Monochromatic Red
                if blues == 5: return False # Monochromatic Blue
                
                if reds == 4:
                    # Must be Blue
                    for v in uncolored:
                        if color[v] == -1:
                            color[v] = 1
                            changed = True
                elif blues == 4:
                    # Must be Red
                    for v in uncolored:
                        if color[v] == -1:
                            color[v] = 0
                            changed = True
        return True

    def backtrack(idx):
        if idx == n:
            return True # All assigned, no conflict found
        
        # Try Red (0)
        orig = color[idx]
        color[idx] = 0
        if propagate(color):
            if backtrack(idx + 1):
                return True
        
        # Try Blue (1)
        color[idx] = 1
        if propagate(color):
            if backtrack(idx + 1):
                return True
        
        color[idx] = orig
        return False

    # Initial propagation
    if not propagate(color):
        return False
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Fallback: The known valid 126-edge construction (Complete 5-uniform on 9 vertices)
    fallback_edges = [list(e) for e in combinations(range(9), 5)]
    best_edges = fallback_edges
    best_n = 9
    best_score = len(best_edges) # We want to minimize this
    
    # Strategy: Search for smaller hypergraphs on n=10, 11, 12
    # Start with dense hypergraphs and remove edges.
    
    for n in [10, 11, 12]:
        if time.time() > deadline - 20: break
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Start with all edges
        current_edges = set(all_edges)
        
        # Check if current is non-2-colorable (it should be for n>=9)
        if _is_2_colorable(n, list(current_edges)):
            continue # Should not happen for complete hypergraph n>=9
        
        # Try to remove edges
        # We need to iterate over edges and try removing them
        # To be efficient, we can randomly sample edges to remove
        # Or iterate through a shuffled list
        
        edges_to_try = list(current_edges)
        random.shuffle(edges_to_try)
        
        removed_count = 0
        for e in edges_to_try:
            if time.time() > deadline - 5: break
            
            # Try removing e
            current_edges.discard(e)
            
            # Check if still non-2-colorable
            if not _is_2_colorable(n, list(current_edges)):
                # Success, keep it removed
                removed_count += 1
                if len(current_edges) < best_score:
                    best_score = len(current_edges)
                    best_n = n
                    best_edges = list(current_edges)
            else:
                # Failed, put it back
                current_edges.add(e)
        
        # If we found a better one, we can try to continue removing more
        # But the single pass might not be optimal.
        # For now, we take the result of this pass.

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
