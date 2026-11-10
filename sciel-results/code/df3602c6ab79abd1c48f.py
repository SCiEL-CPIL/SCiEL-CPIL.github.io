"""
Strategy: Randomized Greedy Construction.
We build the hypergraph incrementally.
1. Start with a small number of vertices (e.g., 10-14).
2. Repeatedly add random 5-edges.
3. After each addition, check if the current hypergraph is 2-colorable.
   - If it is 2-colorable, we might keep the edge if it seems to constrain the solution space significantly, or discard it.
   - Actually, a better heuristic for "not 2-colorable" is to try to satisfy the constraints. 
   - We want a hypergraph that is NOT 2-colorable.
   - So, we add edges. If the hypergraph becomes 2-colorable, we continue adding. 
   - Wait, adding edges makes it HARDER to be 2-colorable. 
   - So we just add random edges until it is no longer 2-colorable.
   - Then we try to remove edges to minimize the count.
   
   This is effectively a "Random Saturation" or "Random Threshold" approach.
   The phase transition for Property B in 5-uniform hypergraphs is around $m \approx c \cdot n^{1-1/4}$? No, for $k$-uniform, it's related to $2^{n/k}$.
   For $k=5$, $n \approx 10-15$, the threshold is low.
   
   Let's refine:
   1. Pick $n$ (e.g., 12).
   2. Generate all possible 5-edges.
   3. Shuffle them.
   4. Add edges one by one.
   5. After each addition, check if the hypergraph is 2-colorable.
   6. If it is NOT 2-colorable, stop adding.
   7. Now we have a non-2-colorable hypergraph. Try to remove edges to minimize the count while keeping it non-2-colorable.
   
   This is different from the seed (which starts with ALL edges) because we start with NONE and add until the threshold is crossed. The resulting core might be sparser or have different structural properties that allow for more aggressive removal.
"""

import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable.
    Returns True if it is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Pre-compute edge masks for speed
    # We use a list of bitmasks for edges
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # State: 
    # assigned: int bitmask of assigned vertices
    # color: dict or list? Let's use two bitmasks: red_mask, blue_mask
    # If a vertex is in red_mask, it's Red. If in blue_mask, it's Blue.
    # If in neither, it's unassigned.
    
    def propagate(red, blue, edges_left_idx):
        """
        Unit propagation.
        Returns (success, red, blue, edges_left_idx)
        If success is False, contradiction.
        """
        changed = True
        while changed:
            changed = False
            # Check edges: if an edge has 4 vertices of one color, the 5th must be the other.
            # If an edge has 5 vertices of one color, contradiction.
            new_edges = []
            for i, mask in enumerate(edge_masks):
                if i not in edges_left_idx:
                    continue
                # Count red and blue vertices in this edge
                r_count = bin(red & mask).count('1')
                b_count = bin(blue & mask).count('1')
                
                if r_count == 5 or b_count == 5:
                    return False, red, blue, edges_left_idx # Contradiction
                
                if r_count == 4:
                    # The 5th vertex must be Blue
                    # Find the unassigned vertex in this edge
                    unassigned_mask = mask & ~(red | blue)
                    # There should be exactly one bit set in unassigned_mask
                    if unassigned_mask == 0:
                        return False, red, blue, edges_left_idx
                    # Set it to Blue
                    blue |= unassigned_mask
                    changed = True
                    # Edge is now satisfied (4R, 1B), remove from active constraints? 
                    # Actually, once satisfied, it doesn't constrain further. 
                    # But wait, if we assign the last vertex, it's satisfied. 
                    # We can remove it from the list of edges to check.
                    # But we are iterating. Let's just mark it satisfied.
                    # For simplicity, we keep it in the list but it won't trigger again?
                    # No, if we keep it, next iteration r_count=4, b_count=1. 
                    # It's stable. 
                    # However, removing satisfied edges speeds up propagation.
                    # Let's filter satisfied edges at the end of the while loop.
                
                elif b_count == 4:
                    # The 5th vertex must be Red
                    unassigned_mask = mask & ~(red | blue)
                    if unassigned_mask == 0:
                        return False, red, blue, edges_left_idx
                    red |= unassigned_mask
                    changed = True
                    
            # Remove satisfied edges from consideration
            # An edge is satisfied if it has both colors.
            # Actually, for unit propagation, we only care about edges that are "one step away" from contradiction or completion.
            # If an edge has 1R and 4U, it's not a unit clause.
            # If an edge has 2R, 2B, 1U, it's not a unit clause.
            # Only 4R, 1U or 4B, 1U are unit clauses.
            # So we don't strictly need to remove satisfied edges, just that the logic above handles the unit cases.
            # However, if an edge becomes satisfied (e.g. 2R, 3B), it will never be a unit clause again.
            # We can filter them out to speed up the loop.
            
            # Optimization: Filter out edges that are already satisfied (contain both colors)
            # Or rather, just keep the logic simple. The loop above is O(E).
            pass
            
        return True, red, blue, edges_left_idx

    def backtrack(red, blue):
        # Propagate first
        ok, red, blue, _ = propagate(red, blue, set(range(len(edges))))
        if not ok:
            return False
            
        # Check if all vertices are assigned
        if (red | blue) == (1 << n) - 1:
            # All assigned. Check if any edge is monochromatic?
            # Propagation should have caught contradictions.
            # If we reach here, it's a valid coloring.
            return True
            
        # Choose an unassigned vertex
        # Heuristic: choose a vertex involved in the most "tight" edges? 
        # For simplicity, just pick the first unassigned one.
        unassigned_mask = ((1 << n) - 1) & ~(red | blue)
        if unassigned_mask == 0:
            return True # Should be caught above
            
        # Pick lowest bit
        v = (unassigned_mask & -unassigned_mask).bit_length() - 1
        
        # Try Red
        if backtrack(red | (1 << v), blue):
            return True
        # Try Blue
        if backtrack(red, blue | (1 << v)):
            return True
            
        return False

    return backtrack(0, 0)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_edges = None
    best_count = float('inf')
    
    # We will try several random instances with different n
    # Known best is 51. We aim for < 51.
    # Let's try n=10, 11, 12, 13, 14.
    
    for n in range(10, 15):
        if time.time() > deadline:
            break
            
        # Generate all possible edges
        all_edges = list(combinations(range(n), 5))
        
        # Shuffle
        random.shuffle(all_edges)
        
        current_edges = []
        
        # Add edges until not 2-colorable
        for e in all_edges:
            if time.time() > deadline:
                break
                
            current_edges.append(e)
            
            # Check if 2-colorable
            # This is the expensive part. 
            # Optimization: Only check every few edges? 
            # No, we need to know when it becomes non-2-colorable.
            # But checking every edge is slow.
            # Heuristic: Check every 5 edges? 
            # If it becomes non-2-colorable, the last few edges are critical.
            
            if len(current_edges) % 5 == 0 or len(current_edges) == len(all_edges):
                if not _is_2_colorable(n, current_edges):
                    # Found a non-2-colorable hypergraph
                    # Now try to remove edges
                    break
        
        if len(current_edges) == 0 or _is_2_colorable(n, current_edges):
            continue # Didn't reach non-2-colorable state
            
        # Now we have a non-2-colorable hypergraph in current_edges
        # Try to minimize it
        # Greedy removal: iterate edges, try to remove, check if still non-2-colorable.
        
        # Sort edges by some heuristic? Maybe random order for removal.
        # Or, remove edges that are "redundant".
        
        edges_to_try = list(current_edges)
        random.shuffle(edges_to_try)
        
        # We will try to remove edges from current_edges
        # Note: current_edges is a list of tuples.
        
        # To speed up, we can keep a set of edge indices or just the list.
        # Since n is small, checking is the bottleneck.
        
        # Let's try a simple greedy removal
        # For each edge in a random order:
        #   if removing it keeps the graph non-2-colorable, remove it.
        
        # However, this is O(E^2 * Cost_Check).
        # Cost_Check is backtracking.
        
        # Let's limit the number of removal attempts per instance.
        max_removal_checks = 500
        checks_done = 0
        
        # We need a mutable list for current_edges
        active_edges = list(current_edges)
        
        # Shuffle active edges for removal order
        random.shuffle(active_edges)
        
        for e in active_edges:
            if time.time() > deadline:
                break
            if checks_done > max_removal_checks:
                break
                
            # Try removing e
            new_edges = [x for x in active_edges if x != e]
            
            if not _is_2_colorable(n, new_edges):
                # Still non-2-colorable, so we can remove e
                active_edges = new_edges
                checks_done += 1
            else:
                # Removing e makes it 2-colorable, so we must keep e
                # Do not increment checks_done? Or do? 
                # It was a check.
                checks_done += 1
        
        if len(active_edges) < best_count:
            best_count = len(active_edges)
            best_edges = active_edges
            
            # If we found something very good, we can stop early?
            # Maybe if we get < 40?
            if best_count <= 40:
                break
                
    if best_edges is None:
        # Fallback to the known good one or the seed?
        # The seed is 126 edges. We should return something valid.
        # If we didn't find anything, return a small valid one?
        # Actually, if the loop didn't find a non-2-colorable one, 
        # it means our random search failed or n was too small.
        # Let's return the complete hypergraph on 9 vertices as a safe fallback, 
        # but that's 126 edges.
        # Or, just return the first non-2-colorable one found in a more deterministic way?
        
        # Fallback: Return the seed's result (126 edges) to ensure validity, 
        # even if score is 1.0.
        from itertools import combinations
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': n, 'edges': [list(e) for e in best_edges]}
