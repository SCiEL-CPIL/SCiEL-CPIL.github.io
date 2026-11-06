import time
import random
from itertools import combinations

def _check_2colorable(n, edges):
    """Check if hypergraph is 2-colorable using bitmask backtracking with unit propagation."""
    if not edges:
        return True
    
    # Precompute edge masks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    num_edges = len(edge_masks)
    all_mask = (1 << n) - 1
    
    # For each vertex, precompute which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, m in enumerate(edge_masks):
        for v in range(n):
            if m & (1 << v):
                vertex_edges[v].append(i)
    
    def solve(pos, c0, c1):
        """Backtracking with unit propagation using bitmasks.
        c0: bitmask of vertices colored 0
        c1: bitmask of vertices colored 1
        Returns True if 2-colorable (found a valid coloring).
        """
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for i, m in enumerate(edge_masks):
                # Count vertices in this edge that are in c0, c1, or unassigned
                in_c0 = m & c0
                in_c1 = m & c1
                in_unassigned = m & ~(c0 | c1)
                
                # If all 5 vertices are assigned
                if in_unassigned == 0:
                    # Check if monochromatic
                    if in_c1 == 0 or in_c0 == 0:
                        return False  # monochromatic edge found
                elif in_unassigned & (in_unassigned - 1) == 0:
                    # Exactly one unassigned vertex
                    unassigned_bit = in_unassigned
                    # Check if all 4 assigned are same color
                    if in_c1 == 0 and in_c0 != 0:
                        # All assigned are color 0, so unassigned must be color 1
                        c1 |= unassigned_bit
                        changed = True
                    elif in_c0 == 0 and in_c1 != 0:
                        # All assigned are color 1, so unassigned must be color 0
                        c0 |= unassigned_bit
                        changed = True
        
        # Check for monochromatic edges after propagation
        for m in edge_masks:
            if m & ~(c0 | c1) == 0:
                if m & c1 == 0 or m & c0 == 0:
                    return False
        
        # All vertices assigned
        if (c0 | c1) == all_mask:
            return True
        
        # Pick next vertex: MRV - vertex in most edges with unassigned neighbors
        unassigned = ~(c0 | c1) & all_mask
        if unassigned == 0:
            return True
        
        # Find vertex with most unassigned neighbors in edges
        best_v = -1
        best_count = -1
        temp = unassigned
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            # Count unassigned vertices in edges containing v
            count = 0
            for ei in vertex_edges[v]:
                m = edge_masks[ei]
                unassigned_in_edge = m & unassigned
                if unassigned_in_edge:
                    count += 1
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v < 0:
            return True
        
        # Try color 0 first, then color 1
        v_bit = 1 << best_v
        if solve(pos + 1, c0 | v_bit, c1):
            return True
        if solve(pos + 1, c0, c1 | v_bit):
            return True
        
        return False
    
    return solve(0, 0, 0)

def _try_remove_edges(n, edges, deadline):
    """Try to remove edges one by one while maintaining non-2-colorability."""
    current = list(edges)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = current[:i] + current[i+1:]
            if not _check_2colorable(n, candidate):
                current = candidate
                changed = True
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_edges = None
    best_n = None
    best_count = 201
    
    # Strategy: Use structured seeds that are likely non-2-colorable,
    # then greedily remove edges.
    
    # Seed 1: Complete 5-uniform hypergraph on 10 vertices
    # C(10,5) = 252 edges, definitely non-2-colorable
    # But we can try smaller structured sets
    
    # Try various n values with random + structured seeds
    random.seed(12345)
    
    for n in range(9, 20):
        if time.time() > deadline - 10:
            break
        
        # Generate multiple random seeds
        for attempt in range(8):
            if time.time() > deadline - 10:
                break
            
            # Mix of random and structured edges
            all_possible = list(combinations(range(n), 5))
            
            # Random subset
            num_random = random.randint(20, min(60, len(all_possible)))
            random.shuffle(all_possible)
            edges = [list(e) for e in all_possible[:num_random]]
            
            # Check if non-2-colorable
            if not _check_2colorable(n, edges):
                # Try to reduce
                reduced = _try_remove_edges(n, edges, deadline)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_edges = reduced
                    best_n = n
                    # More aggressive reduction
                    reduced2 = _try_remove_edges(n, reduced, deadline)
                    if len(reduced2) < best_count:
                        best_count = len(reduced2)
                        best_edges = reduced2
                        best_n = n
    
    # Also try: take a known non-2-colorable structure and add random edges
    # The "truncated" approach: start with dense, remove edges
    for n in range(9, 18):
        if time.time() > deadline - 10:
            break
        
        all_possible = list(combinations(range(n), 5))
        
        # Start with a large random subset
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            random.shuffle(all_possible)
            num_start = min(len(all_possible), random.randint(30, 80))
            edges = [list(e) for e in all_possible[:num_start]]
            
            if not _check_2colorable(n, edges):
                reduced = _try_remove_edges(n, edges, deadline)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_edges = reduced
                    best_n = n
    
    # Fallback: complete hypergraph on 9 vertices (C(9,5)=126 edges)
    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': best_edges}
