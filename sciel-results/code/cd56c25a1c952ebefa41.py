"""Find maximum-edge triangle-free non-3-colorable graph on 16 vertices."""
import time
import random
import sys

def construct():
    deadline = time.time() + 110  # safety margin before 150s limit
    n = 16
    rng = random.Random(42)
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_to_idx = {}
    for idx, e in enumerate(all_edges):
        edge_to_idx[(e[0], e[1])] = idx
        edge_to_idx[(e[1], e[0])] = idx
    
    # Precompute for each edge, the set of edges that would form a triangle with it
    # edge (u,v) conflicts with (u,w) and (v,w) for any w != u,v
    # We'll use bitmasks for adjacency
    # Each vertex has a 16-bit mask of its neighbors
    
    def is_triangle_free(adj_masks):
        """Check if graph represented by adjacency masks is triangle-free."""
        for u in range(n):
            for v in range(u+1, n):
                if adj_masks[u] & (1 << v):
                    if adj_masks[u] & adj_masks[v]:
                        return False
        return True
    
    def is_3_colorable(adj_masks, deadline):
        """Check if graph is 3-colorable using backtracking with bitmasks."""
        # Build adjacency lists
        adj_list = [[] for _ in range(n)]
        for u in range(n):
            mask = adj_masks[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                adj_list[u].append(v)
                mask &= mask - 1
        
        # Sort vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: -len(adj_list[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        colors = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return True
            v = order[idx]
            # Determine forbidden colors from already-colored neighbors
            forbidden = [False] * 3
            for nb in adj_list[v]:
                if colors[nb] >= 0:
                    forbidden[colors[nb]] = True
            
            # Try each color
            for c in range(3):
                if not forbidden[c]:
                    colors[v] = c
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    colors[v] = -1
            return False
        
        result = backtrack(0)
        return result is not None
    
    def is_3_colorable_fast(adj_masks, deadline):
        """Faster 3-colorability check using DSATUR-like heuristic."""
        adj_list = [[] for _ in range(n)]
        for u in range(n):
            mask = adj_masks[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                adj_list[u].append(v)
                mask &= mask - 1
        
        colors = [-1] * n
        uncolored = set(range(n))
        
        def get_saturation(v):
            """Number of distinct colors used by neighbors of v."""
            s = set()
            for nb in adj_list[v]:
                if colors[nb] >= 0:
                    s.add(colors[nb])
            return len(s)
        
        def backtrack():
            if time.time() > deadline:
                return None
            if not uncolored:
                return True
            
            # Choose uncolored vertex with max saturation, then max degree
            best_v = None
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat = get_saturation(v)
                deg = len(adj_list[v])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
            
            v = best_v
            uncolored.remove(v)
            
            forbidden = [False] * 3
            for nb in adj_list[v]:
                if colors[nb] >= 0:
                    forbidden[colors[nb]] = True
            
            for c in range(3):
                if not forbidden[c]:
                    colors[v] = c
                    result = backtrack()
                    if result is not None:
                        return result
                    colors[v] = -1
            
            uncolored.add(v)
            return False
        
        return backtrack() is not None
    
    def edge_count(adj_masks):
        count = 0
        for u in range(n):
            mask = adj_masks[u]
            while mask:
                count += 1
                mask &= mask - 1
        return count // 2
    
    def masks_to_edges(adj_masks):
        edges = []
        for u in range(n):
            mask = adj_masks[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask &= mask - 1
        return edges
    
    def edges_to_masks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    # Strategy: Use local search / simulated annealing
    # State: adjacency masks (triangle-free)
    # Objective: maximize edge count subject to non-3-colorability
    # 
    # We'll use a two-phase approach:
    # Phase 1: Find a good non-3-colorable triangle-free graph
    # Phase 2: Try to add more edges while maintaining both properties
    
    best_masks = None
    best_count = 0
    
    # Start with several random initializations
    # Use the Clebsch graph as one starting point
    clebsch_masks = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            diff = i ^ j
            if bin(diff).count('1') == 2:
                clebsch_masks[i] |= (1 << j)
                clebsch_masks[j] |= (1 << i)
    
    # Generate random maximal triangle-free graphs
    def random_maximal_triangle_free():
        """Generate a random maximal triangle-free graph."""
        masks = [0] * n
        # Random order of vertices
        vertex_order = list(range(n))
        rng.shuffle(vertex_order)
        
        for u in vertex_order:
            # Decide which later vertices to connect to
            for v_idx in range(vertex_order.index(u) + 1, n):
                v = vertex_order[v_idx]
                # Check if adding (u,v) creates a triangle
                # Need: no common neighbor of u and v
                if not (masks[u] & masks[v]):
                    if rng.random() < 0.5:  # random probability
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
        
        # Try to add more edges greedily
        for u in range(n):
            for v in range(u+1, n):
                if not (masks[u] & (1 << v)):
                    if not (masks[u] & masks[v]):
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
        
        return masks
    
    # Local search: try to add edges
    def try_add_edges(masks, deadline):
        """Try to add edges while maintaining triangle-free and non-3-colorable."""
        current_masks = masks[:]
        
        # Get candidate edges
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if not (current_masks[u] & (1 << v)):
                    if not (current_masks[u] & current_masks[v]):
                        candidates.append((u, v))
        
        # Shuffle candidates
        rng.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Try adding edge (u,v)
            new_masks = current_masks[:]
            new_masks[u] |= (1 << v)
            new_masks[v] |= (1 << u)
            
            # Check triangle-free
            if not is_triangle_free(new_masks):
                continue
            
            # Check non-3-colorable
            if not is_3_colorable_fast(new_masks, deadline):
                current_masks = new_masks
        
        return current_masks
    
    # Local search: try edge swaps
    def try_edge_swap(masks, deadline):
        """Try to swap an edge for a better one."""
        current_masks = masks[:]
        
        # Get current edges
        current_edges = []
        for u in range(n):
            mask = current_masks[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                if u < v:
                    current_edges.append((u, v))
                mask &= mask - 1
        
        if not current_edges:
            return current_masks
        
        # Get candidate edges not in current graph
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if not (current_masks[u] & (1 << v)):
                    if not (current_masks[u] & current_masks[v]):
                        candidates.append((u, v))
        
        if not candidates:
            return current_masks
        
        # Try swapping: remove one edge, add another
        rng.shuffle(current_edges)
        rng.shuffle(candidates)
        
        for remove_edge in current_edges[:10]:
            if time.time() > deadline:
                break
            
            u1, v1 = remove_edge
            new_masks = current_masks[:]
            new_masks[u1] &= ~(1 << v1)
            new_masks[v1] &= ~(1 << u1)
            
            for add_edge in candidates[:10]:
                if time.time() > deadline:
                    break
                
                u2, v2 = add_edge
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                
                # Check if adding (u2,v2) creates a triangle
                if new_masks[u2] & new_masks[v2]:
                    continue
                
                trial_masks = new_masks[:]
                trial_masks[u2] |= (1 << v2)
                trial_masks[v2] |= (1 << u2)
                
                if not is_triangle_free(trial_masks):
                    continue
                
                if not is_3_colorable_fast(trial_masks, deadline):
                    return trial_masks
        
        return current_masks
    
    # Main search loop
    # Start with multiple initializations
    initializations = [clebsch_masks]
    for _ in range(20):
        initializations.append(random_maximal_triangle_free())
    
    for init_masks in initializations:
        if time.time() > deadline:
            break
        
        # First, check if this initialization is non-3-colorable
        if is_3_colorable_fast(init_masks, deadline):
            # Try to make it non-3-colorable by removing edges
            # This is harder; skip for now or try random removals
            continue
        
        current_masks = init_masks
        current_count = edge_count(current_masks)
        
        if current_count > best_count:
            best_masks = current_masks[:]
            best_count = current_count
        
        # Iteratively try to add edges
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            new_masks = try_add_edges(current_masks, deadline)
            new_count = edge_count(new_masks)
            
            if new_count > current_count:
                current_masks = new_masks
                current_count = new_count
                if current_count > best_count:
                    best_masks = current_masks[:]
                    best_count = current_count
            else:
                # Try edge swaps
                new_masks = try_edge_swap(current_masks, deadline)
                new_count = edge_count(new_masks)
                if new_count > current_count:
                    current_masks = new_masks
                    current_count = new_count
                    if current_count > best_count:
                        best_masks = current_masks[:]
                        best_count = current_count
                else:
                    break
        
        # Also try starting from this and doing more aggressive search
        if time.time() > deadline:
            break
    
    # If we didn't find anything, try harder
    if best_masks is None or best_count == 0:
        # Use Grötzsch graph as fallback
        groetzsch_edges = []
        for i in range(5):
            groetzsch_edges.append([i, (i+1)%5])
        for i in range(5):
            j = (i+1)%5
            groetzsch_edges.append([5+i, j])
            groetzsch_edges.append([5+j, i])
        for i in range(5):
            groetzsch_edges.append([10, 5+i])
        
        best_masks = edges_to_masks(groetzsch_edges)
        best_count = len(groetzsch_edges)
    
    # Final verification
    if best_masks is not None:
        assert is_triangle_free(best_masks), "Result should be triangle-free"
        assert not is_3_colorable_fast(best_masks, deadline), "Result should not be 3-colorable"
    
    return masks_to_edges(best_masks) if best_masks else []

