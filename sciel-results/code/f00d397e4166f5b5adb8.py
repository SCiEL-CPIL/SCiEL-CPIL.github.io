"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Robust 2-colorability check.
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Preprocess edges: remove duplicates, sort vertices in edge
    edges = [tuple(sorted(e)) for e in edges]
    edges = list(set(edges))
    if not edges:
        return True

    # For small n, bitmask brute force is faster
    if n <= 16:
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        full_mask = (1 << n) - 1
        # Check all 2^n colorings
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                ones = bin(em & mask).count('1')
                # If ones == 5, all vertices in edge are 1
                # If ones == 0, all vertices in edge are 0
                if ones == 5 or ones == 0:
                    valid = False
                    break
            if valid:
                return True
        return False

    # For larger n, use backtracking
    # Map vertices to indices 0..n-1
    color = [-1] * n
    
    # Precompute edge vertex lists for faster access
    edge_verts = [list(e) for e in edges]
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for edge in edge_verts:
                c0 = 0
                c1 = 0
                un = 0
                un_v = -1
                for v in edge:
                    if color[v] == 0:
                        c0 += 1
                    elif color[v] == 1:
                        c1 += 1
                    else:
                        un += 1
                        un_v = v
                
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and un == 1:
                    color[un_v] = 1
                    changed = True
                elif c1 == 4 and un == 1:
                    color[un_v] = 0
                    changed = True
        return True
    
    def backtrack():
        # Find an uncolored vertex
        # Heuristic: pick vertex with highest degree in uncolored subgraph? 
        # For simplicity, pick first uncolored. 
        # Better: pick vertex involved in most constrained edges.
        # Let's stick to simple order for now, but propagate heavily.
        
        # Check if all colored
        for v in range(n):
            if color[v] == -1:
                break
        else:
            return True
            
        v = v # The vertex we found
        
        # Try color 0
        color[v] = 0
        if propagate():
            if backtrack():
                return True
        color[v] = -1
        
        # Try color 1
        color[v] = 1
        if propagate():
            if backtrack():
                return True
        color[v] = -1
        
        return False
        
    return backtrack()

def _minimize_edges(n, edges, deadline):
    """
    Core extraction: iteratively remove edges that do not contribute to non-2-colorability.
    """
    edges = [tuple(e) for e in edges]
    edges = list(set(edges))
    
    # Sort edges by some heuristic? Random order is fine for core extraction if we iterate until stable.
    # We will do multiple passes.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias? Or keep order.
        # Let's iterate through a copy to allow removals
        current_edges = edges[:]
        for i in range(len(current_edges)):
            if time.time() > deadline:
                break
            e = current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            # Check if removing e makes it 2-colorable
            if _is_2_colorable(n, test_edges):
                # e is essential, keep it
                pass
            else:
                # e is not essential, remove it
                edges = test_edges
                improved = True
                # Restart loop as indices changed
                break
    return [list(e) for e in edges]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    best_n = 0
    best_edges = None
    best_len = float('inf')
    
    # Strategy: Try different n values, generate random dense hypergraphs, minimize.
    # Known: m(5) <= 51. We aim to find something close to or below 51.
    # n=8 gives max 56 edges. K_8^5 is non-2-colorable.
    # n=9 gives max 126 edges.
    # n=10 gives max 252 edges.
    
    # 1. Try n=8 (Small, fast checks)
    if time.time() < deadline - 10:
        n = 8
        all_edges_8 = [list(e) for e in combinations(range(n), 5)]
        # K_8^5 is non-2-colorable
        if not _is_2_colorable(n, all_edges_8):
            edges8 = _minimize_edges(n, all_edges_8, deadline - 5)
            if not _is_2_colorable(n, edges8) and len(edges8) < best_len:
                best_edges = edges8[:]
                best_len = len(edges8)
                best_n = n
    
    # 2. Try n=9
    if time.time() < deadline - 20:
        n = 9
        all_edges_9 = list(combinations(range(n), 5))
        # Random restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            # Start with a random subset of ~80 edges
            initial_size = min(80, len(all_edges_9))
            random_edges = random.sample(all_edges_9, initial_size)
            edges9 = [list(e) for e in random_edges]
            
            # Ensure non-2-colorable
            # If it is 2-colorable, add random edges until it isn't or we run out
            while _is_2_colorable(n, edges9) and time.time() < deadline - 10:
                current_set = set(tuple(e) for e in edges9)
                remaining = [e for e in all_edges_9 if e not in current_set]
                if not remaining:
                    break
                edges9.append(list(random.choice(remaining)))
            
            if not _is_2_colorable(n, edges9):
                edges9_min = _minimize_edges(n, edges9, deadline - 5)
                if not _is_2_colorable(n, edges9_min) and len(edges9_min) < best_len:
                    best_edges = edges9_min[:]
                    best_len = len(edges9_min)
                    best_n = n
    
    # 3. Try n=10
    if time.time() < deadline - 30:
        n = 10
        all_edges_10 = list(combinations(range(n), 5))
        # Random restarts
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            # Start with a random subset of ~100 edges
            initial_size = min(100, len(all_edges_10))
            random_edges = random.sample(all_edges_10, initial_size)
            edges10 = [list(e) for e in random_edges]
            
            # Ensure non-2-colorable
            while _is_2_colorable(n, edges10) and time.time() < deadline - 15:
                current_set = set(tuple(e) for e in edges10)
                remaining = [e for e in all_edges_10 if e not in current_set]
                if not remaining:
                    break
                edges10.append(list(random.choice(remaining)))
            
            if not _is_2_colorable(n, edges10):
                edges10_min = _minimize_edges(n, edges10, deadline - 5)
                if not _is_2_colorable(n, edges10_min) and len(edges10_min) < best_len:
                    best_edges = edges10_min[:]
                    best_len = len(edges10_min)
                    best_n = n
    
    # Fallback
    if best_edges is None:
        # Return a known valid instance: K_8^5
        best_n = 8
        best_edges = [list(e) for e in combinations(range(8), 5)]
    
    # Final check
    if _is_2_colorable(best_n, best_edges):
        # This should not happen if logic is correct, but safety net
        # Return K_8^5 which is definitely non-2-colorable
        best_n = 8
        best_edges = [list(e) for e in combinations(range(8), 5)]

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
