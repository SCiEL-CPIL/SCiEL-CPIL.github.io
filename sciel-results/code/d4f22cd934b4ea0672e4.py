"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable_fast(n, edges):
    """
    Check if hypergraph is 2-colorable using optimized DPLL with bitset unit propagation.
    Returns True if colorable, False otherwise.
    """
    if not edges:
        return True
        
    # Convert edges to bitmasks for fast intersection
    edge_masks = [0] * len(edges)
    for i, e in enumerate(edges):
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks[i] = m
            
    # Incidence list: vertex -> list of edge indices
    incidence = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            incidence[v].append(i)
            
    # State:
    # assignments: dict {vertex: 0/1}
    # unassigned_mask: bitmask of unassigned vertices
    # edge_state: list of (mask_0, mask_1) for each edge? 
    # No, better: maintain for each edge the set of unassigned vertices and counts of 0/1.
    # Or even better: maintain "active" edges (those not yet satisfied/violated).
    
    # Let's use a recursive solver with backtracking.
    # To speed up, we use bitmasks for the current partial assignment.
    
    # We'll represent the state as:
    # assigned_mask: bitmask of assigned vertices
    # color_mask: bitmask of vertices colored 1 (True)
    # 
    # An edge is violated if:
    # 1. All vertices in edge are assigned and all are 0 (edge_mask & ~color_mask == edge_mask and edge_mask & assigned_mask == edge_mask)
    #    Wait, if all assigned and all 0: (edge_mask & assigned_mask) == edge_mask AND (edge_mask & color_mask) == 0
    # 2. All vertices in edge are assigned and all are 1: (edge_mask & assigned_mask) == edge_mask AND (edge_mask & color_mask) == edge_mask
    
    # An edge is "forced" (unit propagation) if:
    # 1. 4 vertices assigned to 0, 1 unassigned -> that unassigned MUST be 1.
    #    Condition: (edge_mask & assigned_mask) has 4 bits, and those 4 are in ~color_mask.
    #    Let unassigned_in_edge = edge_mask & ~assigned_mask.
    #    If popcount(unassigned_in_edge) == 1:
    #       Let v be that vertex.
    #       If all other 4 are 0, then v must be 1.
    #       If all other 4 are 1, then v must be 0.
        
    # This per-edge check is O(E) per node. With E~50 and depth ~10-20, it's feasible.
    # But we can do better with precomputed structures or incremental updates.
    # Given the small size, O(E) per node is likely fine if the branching factor is low due to propagation.
    
    # Let's implement a simple but effective recursive solver.
    
    # Precompute popcount for masks? Python int.bit_count() is fast.
    
    def solve(assigned_mask, color_mask):
        # Check for violations and propagations
        # We need to iterate through edges.
        # To make it faster, we can maintain a list of "active" edges, but for simplicity 
        # and given small E, we iterate all edges.
        
        forced_var = -1
        forced_val = -1
        
        for i in range(len(edges)):
            em = edge_masks[i]
            
            # Mask of assigned vertices in this edge
            assigned_in_edge = em & assigned_mask
            # Mask of unassigned vertices in this edge
            unassigned_in_edge = em & ~assigned_mask
            
            # If no unassigned, check if violated
            if unassigned_in_edge == 0:
                # Check if monochromatic
                # Count 1s: (em & color_mask).bit_count()
                # If 0 or 5, it's violated.
                ones = (em & color_mask).bit_count()
                if ones == 0 or ones == 5:
                    return False
                # Else, this edge is satisfied (has mix). It doesn't constrain further.
                continue
            
            # If some unassigned, check for unit propagation
            # If exactly 1 unassigned
            if unassigned_in_edge & (unassigned_in_edge - 1) == 0:
                # It is a power of 2, so single bit
                v_bit = unassigned_in_edge
                # The other 4 are assigned
                # Check if all other 4 are 0
                # Other assigned mask = assigned_in_edge
                # If (assigned_in_edge & color_mask) == 0, all are 0 -> v must be 1
                if (assigned_in_edge & color_mask) == 0:
                    # All assigned are 0
                    if forced_var != -1:
                        # Conflict if we already forced a different var or value
                        if forced_var == v_bit and forced_val == 0:
                            pass # Consistent
                        elif forced_var == v_bit and forced_val == 1:
                            return False # Conflict
                        else:
                            # Multiple propagations? This logic is getting messy.
                            # Better to collect all propagations and check for consistency.
                            pass
                    forced_var = v_bit
                    forced_val = 1
                else:
                    # Check if all other 4 are 1
                    # This means assigned_in_edge is subset of color_mask
                    if (assigned_in_edge & ~color_mask) == 0:
                        # All assigned are 1
                        if forced_var != -1:
                            if forced_var == v_bit and forced_val == 1:
                                pass
                            elif forced_var == v_bit and forced_val == 0:
                                return False
                            else:
                                pass
                        forced_var = v_bit
                        forced_val = 0
                        # Note: If assigned_in_edge has mixed colors, no propagation.
                        # But wait, if 2 are 0 and 2 are 1, no propagation.
                        # The condition (assigned_in_edge & color_mask) == 0 handles "all 0".
                        # The condition (assigned_in_edge & ~color_mask) == 0 handles "all 1".
                        # If neither, no propagation.
                    else:
                        # Mixed, no propagation
                        pass
                # End if unassigned is 1
            # End if
            
        # If we found a forced variable
        if forced_var != -1:
            # Apply propagation
            if forced_val == 1:
                new_color_mask = color_mask | forced_var
            else:
                new_color_mask = color_mask # forced_var is 0, so not in color_mask
            new_assigned_mask = assigned_mask | forced_var
            return solve(new_assigned_mask, new_color_mask)
            
        # No propagation. Need to branch.
        # Heuristic: Pick a variable that appears in the most "critical" edges?
        # Or just pick the first unassigned vertex?
        # Better: Pick a vertex from an edge that has the most unassigned vertices but is "close" to violation?
        # Simple heuristic: Pick the unassigned vertex with highest degree in the remaining subgraph?
        # Or just pick the lowest index unassigned vertex for determinism and speed.
        
        # Let's pick the unassigned vertex with the most constraints (edges containing it that are not yet satisfied).
        # Actually, for small n, picking the first unassigned is often fine if propagation is strong.
        
        # Find unassigned vertices
        unassigned_all = (1 << n) - 1
        unassigned_vars = unassigned_all & ~assigned_mask
        
        if unassigned_vars == 0:
            return True
            
        # Pick a variable. Let's try to pick one involved in an edge with fewest unassigned?
        # Or just the first one.
        # To improve, let's pick a variable that is in an edge where the other 4 are assigned? 
        # No, that would have been propagated.
        # So all edges have either 0 or >=2 unassigned? No, 1 unassigned was propagated.
        # So all edges have 0 or >=2 unassigned.
        
        # Let's pick the vertex with the highest degree in the original graph.
        # Precomputed degree?
        # Or just iterate.
        
        # Simple: pick the lowest index unassigned vertex.
        # This might be slow if the search space is large.
        # Let's try a heuristic: pick a vertex from an edge that has the maximum number of unassigned vertices?
        # No, that's the opposite. We want to constrain quickly.
        # Pick a vertex from an edge that has the MINIMUM number of unassigned vertices (>=2).
        
        min_unassigned = 99
        best_edge_idx = -1
        for i in range(len(edges)):
            em = edge_masks[i]
            unassigned_in_edge = em & ~assigned_mask
            cnt = unassigned_in_edge.bit_count()
            if 0 < cnt < min_unassigned:
                min_unassigned = cnt
                best_edge_idx = i
                if cnt == 2:
                    break # Can't do better than 2 (since 1 is propagated)
                    
        if best_edge_idx != -1:
            em = edge_masks[best_edge_idx]
            unassigned_in_edge = em & ~assigned_mask
            # Pick one of these vertices
            # Let's pick the first one
            v_bit = unassigned_in_edge & -unassigned_in_edge # lowest set bit
        else:
            # All edges are satisfied? Or no edges?
            # If no edges constrain, any assignment works.
            # But we must assign all? No, if no edges, it's colorable.
            # But we only reach here if unassigned_vars != 0.
            # If best_edge_idx is -1, it means all edges have 0 unassigned (satisfied) or we have no edges?
            # If all edges are satisfied, we can stop?
            # Wait, if an edge has 0 unassigned, it was checked for violation.
            # So if we get here, all edges are either satisfied or we have no edges to check?
            # Actually, if an edge has 0 unassigned, it's satisfied (not violated).
            # So if all edges have 0 unassigned, the partial assignment is extendable?
            # Yes, because no edge is violated, and no edge needs propagation.
            # The remaining unassigned vertices don't affect any edge?
            # Wait, if a vertex is unassigned, it might be in an edge.
            # If it's in an edge, that edge has unassigned vertices.
            # So if best_edge_idx is -1, it means no edge has any unassigned vertices.
            # This implies all vertices in all edges are assigned.
            # So unassigned_vars must be 0?
            # If there are unassigned vertices not in any edge, they don't matter.
            # So we can return True.
            return True

        # Branch
        # Try 0
        new_assigned_mask = assigned_mask | v_bit
        # color_mask stays same for 0
        if solve(new_assigned_mask, color_mask):
            return True
            
        # Try 1
        new_color_mask = color_mask | v_bit
        if solve(new_assigned_mask, new_color_mask):
            return True
            
        return False

    return solve(0, 0)

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    # Base construction: 51 edges on 9 vertices?
    # The problem states best known is 51.
    # We don't know the exact 51 edges. 
    # Strategy: Start with complete graph on 9 vertices (126 edges).
    # Randomly remove edges while maintaining non-2-colorability.
    # This is a "randomized greedy removal" or "local search" approach.
    
    # To get a good starting point, we can try to find a minimal non-2-colorable subgraph 
    # by starting from a random set of ~50-60 edges and adding/removing.
    
    # Better: Start with all 126 edges.
    # Shuffle.
    # Iterate: try to remove edge i. If still non-2-colorable, remove it.
    # This greedy approach gets stuck in local minima (often around 50-60).
    
    # To escape, we can use "simulated annealing" style:
    # 1. Greedy removal to get a small set S.
    # 2. Randomly swap an edge in S with an edge not in S.
    # 3. If the new set is non-2-colorable and has same size (or smaller if we remove), keep it.
    # 4. Repeat.
    
    # But we need to be fast.
    # Let's stick to a robust greedy removal with multiple restarts if time permits.
    
    n = 9
    all_edges = list(combinations(range(n), 5))
    
    # Verify complete graph is non-2-colorable (it is)
    
    best_edges = None
    best_size = 126
    
    # Multiple restarts with different shuffles
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        current_edges = all_edges[:]
        random.shuffle(current_edges)
        
        # Greedy removal
        i = 0
        while i < len(current_edges) and time.time() < deadline - 5:
            # Test if removing current_edges[i] makes it colorable
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            # Optimization: If test_edges size is small, check is fast.
            if _is_2_colorable_fast(n, test_edges):
                # Became colorable, must keep
                i += 1
            else:
                # Still non-colorable, remove
                current_edges.pop(i)
                # Do not increment i
        
        # Check if this result is better
        if len(current_edges) < best_size:
            best_size = len(current_edges)
            best_edges = current_edges[:]
            
        # Optional: Try to improve best_edges with local search?
        # Given time constraints, maybe just do more restarts.
        
    return {'n': n, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
