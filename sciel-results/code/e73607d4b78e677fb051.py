import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable (i.e., there exists a coloring 
    such that no edge is monochromatic).
    Returns False if it is NOT 2-colorable (i.e., every coloring has a monochromatic edge).
    
    Wait, the task says: "The hypergraph must have no proper 2-colouring".
    "no colouring of the vertices leaves every edge with both colours".
    This means we are looking for a hypergraph where EVERY 2-coloring has at least one monochromatic edge.
    So `is_2_colorable` should return True if there EXISTS a coloring with no monochromatic edges.
    We want a hypergraph where `is_2_colorable` returns False.
    """
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    
    # Precompute edge indices for faster access
    edge_indices = [list(e) for e in edges]
    
    def propagate(colors):
        """
        Apply unit propagation. Returns False if contradiction, True if consistent.
        """
        changed = True
        while changed:
            changed = False
            for e in edge_indices:
                # Count colors in this edge
                c0 = 0
                c1 = 0
                unassigned = -1
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned = v
                        # If more than 1 unassigned, we can't force a value yet via unit propagation
                        # unless one side is already "full"
                        if c0 == 4:
                            if colors[v] == 1:
                                return False # Contradiction: 4 zeros, 1 one -> monochromatic? No, 4 zeros and 1 one is NOT monochromatic.
                                # Wait. Monochromatic means ALL 5 are same.
                                # If c0 == 4 and the last is 1, the edge has 4 zeros and 1 one. It is NOT monochromatic.
                                # So no constraint violation.
                                # The constraint is: NOT (c0 == 5) and NOT (c1 == 5).
                                # If c0 == 4 and unassigned is v, v MUST be 1 to avoid c0 becoming 5.
                            if colors[v] == -1:
                                colors[v] = 1
                                changed = True
                        if c1 == 4:
                            if colors[v] == 0:
                                return False
                            if colors[v] == -1:
                                colors[v] = 0
                                changed = True
                        # If c0 == 4 and c1 == 1? No, sum is 5.
                        # If c0 == 4, the 5th MUST be 1.
                        # If c1 == 4, the 5th MUST be 0.
                        # If c0 == 5 or c1 == 5, it's monochromatic -> Invalid.
                        if c0 == 5 or c1 == 5:
                            return False
        return True

    def backtrack(idx):
        if idx == n:
            return True # All assigned, no contradictions found during propagation
        
        # Try color 0
        if colors[idx] == -1:
            colors[idx] = 0
            if propagate(colors):
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
            
            # Try color 1
            colors[idx] = 1
            if propagate(colors):
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
            
            return False
        else:
            # Already assigned by propagation
            if propagate(colors):
                return backtrack(idx + 1)
            return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Start with K9^5
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Verify it's not 2-colorable
    if is_2_colorable(n, current_edges):
        # This shouldn't happen for K9^5, but just in case
        return {'n': n, 'edges': current_edges}
        
    # Greedy removal
    # We want to minimize edges.
    # We shuffle to get different results
    random.seed(42) # Fixed seed for reproducibility, or use time? Task doesn't specify.
    
    best_edges = current_edges[:]
    
    while time.time() < deadline:
        # Shuffle current edges to try different removal orders
        random.shuffle(current_edges)
        removed_any = False
        new_edges = current_edges[:]
        
        for i in range(len(new_edges)):
            if time.time() > deadline:
                break
                
            edge_to_remove = new_edges[i]
            
            # Try removing it
            test_edges = new_edges[:i] + new_edges[i+1:]
            
            # Check if still non-2-colorable
            # i.e., is_2_colorable returns False
            if not is_2_colorable(n, test_edges):
                # It is still non-2-colorable, so we can remove this edge
                new_edges = test_edges
                removed_any = True
                # Update best if smaller
                if len(new_edges) < len(best_edges):
                    best_edges = new_edges[:]
            
        # If no edges were removed in this pass, we might be stuck in a local minimum
        # But since we shuffle, we might try different orders in the next pass.
        # However, greedy removal is deterministic if the order is fixed.
        # Shuffling helps escape some local minima if the "removability" depends on order?
        # Actually, if an edge is removable, it is removable regardless of other edges?
        # No. Removing edge A might make edge B removable, but removing B first might make A non-removable?
        # Yes, the property is not independent.
        
        if not removed_any:
            # No improvement in this full pass.
            # We can break or try a different strategy.
            # For now, break to avoid wasting time if we're stuck.
            # Or, we can try to swap edges? No, we just want to minimize count.
            break
            
        current_edges = new_edges[:]

    return {'n': n, 'edges': best_edges}
