"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable_bitmask(n, edges_mask):
    """
    Fast backtracking with unit propagation for 5-uniform hypergraph 2-colorability.
    edges_mask: list of integers, each an n-bit mask.
    Returns True if 2-colorable (every edge has both colours in some assignment),
    False if not (i.e., there is no proper 2-colouring).
    
    We model: a proper 2-colouring means every edge has at least one 0 and one 1.
    Equivalently, NOT 2-colorable means: for every assignment of {0,1} to vertices,
    some edge is monochromatic.
    
    We search for an assignment where NO edge is monochromatic.
    If we find one -> 2-colorable (return True).
    If search exhausts -> not 2-colorable (return False).
    """
    m = len(edges_mask)
    if m == 0:
        return True
    
    # Precompute for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edges_mask):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
    
    # State: assignment array (-1 = unassigned, 0, 1)
    # We use iterative deepening / backtracking with unit propagation.
    # An edge is "monochromatic" if all its vertices are assigned and all same colour.
    # An edge is "forced" if 4 of its 5 vertices are assigned to the same colour -> 5th must be opposite.
    
    # For speed, track for each edge: count of 0s and 1s assigned.
    # If count0 == 5 or count1 == 5 -> contradiction (edge is mono, invalid for 2-coloring)
    # If count0 == 4 -> 5th vertex must be 1 (unit propagation)
    # If count1 == 4 -> 5th vertex must be 0 (unit propagation)
    
    # We'll use a recursive backtracking with memoization on (assigned_mask, assignment_pattern)
    # Actually, let's use a simpler approach: track per-edge counts.
    
    # For n up to 26, we can use a bitmask for assigned vertices.
    # assignment: two bitmasks (color0_mask, color1_mask)
    
    def backtrack(color0, color1, unassigned):
        """
        color0: bitmask of vertices assigned 0
        color1: bitmask of vertices assigned 1
        unassigned: bitmask of unassigned vertices
        Returns True if a valid 2-coloring exists extending this assignment.
        """
        if unassigned == 0:
            # Check all edges are not monochromatic
            for em in edges_mask:
                c0 = em & color0
                c1 = em & color1
                if c0 == em or c1 == em:
                    return False
            return True
        
        # Unit propagation: check edges that are almost fully assigned
        # For each edge, count assigned 0s and 1s
        # If an edge has 4 zeros and 1 unassigned -> that vertex must be 1
        # If an edge has 4 ones and 1 unassigned -> that vertex must be 0
        # If an edge has 5 of one colour -> contradiction
        
        forced0 = 0  # vertices that must be 0
        forced1 = 0  # vertices that must be 1
        
        for em in edges_mask:
            e0 = em & color0
            e1 = em & color1
            e_un = em & unassigned
            cnt0 = bin(e0).count('1')
            cnt1 = bin(e1).count('1')
            cnt_un = bin(e_un).count('1')
            
            if cnt0 + cnt1 + cnt_un != 5:
                continue  # shouldn't happen
            
            if cnt0 == 5 or cnt1 == 5:
                return False  # contradiction: edge is monochromatic
            
            if cnt0 == 4 and cnt_un == 1:
                # The one unassigned vertex in this edge must be 1
                forced1 |= e_un
            elif cnt1 == 4 and cnt_un == 1:
                # The one unassigned vertex in this edge must be 0
                forced0 |= e_un
        
        # Apply forced assignments
        if forced0 & forced1:
            return False  # contradiction
        
        if forced0:
            color0 |= forced0
            unassigned &= ~forced0
            # Recurse to propagate further
            return backtrack(color0, color1, unassigned)
        
        if forced1:
            color1 |= forced1
            unassigned &= ~forced1
            return backtrack(color0, color1, unassigned)
        
        # Choose a vertex to branch on
        # Heuristic: pick vertex in most constrained edge
        # For simplicity, pick the lowest unassigned vertex
        v = (unassigned & -unassigned).bit_length() - 1
        
        # Try colour 0
        if backtrack(color0 | (1 << v), color1, unassigned & ~(1 << v)):
            return True
        # Try colour 1
        if backtrack(color0, color1 | (1 << v), unassigned & ~(1 << v)):
            return True
        
        return False
    
    return backtrack(0, 0, (1 << n) - 1)


def _remove_edge_if_colorable(n, edges_list, edges_mask):
    """
    Try to remove an edge. If the remaining hypergraph is still not 2-colorable,
    remove it. Returns (new_edges_list, new_edges_mask, removed).
    """
    m = len(edges_list)
    if m == 0:
        return edges_list, edges_mask, False
    
    # Try removing edges in random order, prefer edges that are "redundant"
    # For efficiency, try a few random edges
    order = list(range(m))
    random.shuffle(order)
    
    for idx in order:
        new_mask = edges_mask[:idx] + edges_mask[idx+1:]
        if not _is_2_colorable_bitmask(n, new_mask):
            # Still not 2-colorable, so we can remove this edge
            new_list = edges_list[:idx] + edges_list[idx+1:]
            return new_list, new_mask, True
    
    return edges_list, edges_mask, False


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_result = None
    best_edges_count = float('inf')
    
    # Start from the known 51-edge construction if we had it, 
    # but we'll build from scratch using random accumulation.
    
    # Strategy: 
    # 1. Start with a random set of edges on a small n.
    # 2. Check if not 2-colorable.
    # 3. Try to remove edges.
    # 4. Try different n values.
    
    # Known: best is 51 edges. Let's try to find something close.
    # We'll try n from 10 to 18, since very small n might not support non-2-colorable
    # with few edges, and very large n needs more edges.
    
    for n in [12, 11, 13, 10, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Start with a random subset of edges
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Start with a moderate number of edges
            initial_count = min(len(all_edges), 30 + random.randint(0, 20))
            edges_list = [list(e) for e in all_edges[:initial_count]]
            edges_mask = [sum(1 << v for v in e) for e in edges_list]
            
            # Check if already not 2-colorable
            if not _is_2_colorable_bitmask(n, edges_mask):
                # Try to minimize
                improved = True
                while improved and time.time() < deadline - 5:
                    edges_list, edges_mask, removed = _remove_edge_if_colorable(n, edges_list, edges_mask)
                    improved = removed
                
                if len(edges_list) < best_edges_count:
                    best_edges_count = len(edges_list)
                    best_result = {'n': n, 'edges': edges_list}
            else:
                # Add more edges until not 2-colorable
                remaining = all_edges[initial_count:]
                random.shuffle(remaining)
                for e in remaining:
                    if time.time() > deadline - 5:
                        break
                    edges_list.append(list(e))
                    edges_mask.append(sum(1 << v for v in e))
                    if len(edges_mask) >= 80:  # Cap to avoid too many
                        break
                    if not _is_2_colorable_bitmask(n, edges_mask):
                        # Found non-2-colorable, try to minimize
                        improved = True
                        while improved and time.time() < deadline - 5:
                            edges_list, edges_mask, removed = _remove_edge_if_colorable(n, edges_list, edges_mask)
                            improved = removed
                        
                        if len(edges_list) < best_edges_count:
                            best_edges_count = len(edges_list)
                            best_result = {'n': n, 'edges': edges_list}
                        break
    
    # If we found a good result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return the complete graph on 9 vertices (126 edges)
    # This is guaranteed to be not 2-colorable
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
