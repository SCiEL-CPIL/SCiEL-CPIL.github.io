import time
import random
import itertools
import sys

def _is_2_colorable_fast(n, edges):
    """
    Fast check for 2-colorability of a 5-uniform hypergraph.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation and bitmask representation.
    """
    if not edges:
        return True
        
    # Pre-compute edge masks and vertex degrees
    # For n <= 26, we can use Python integers as bitmasks.
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Vertex degrees (number of edges containing the vertex)
    deg = [0] * n
    for e in edges:
        for v in e:
            deg[v] += 1
            
    # Map vertex to list of edge indices it belongs to
    v_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_to_edges[v].append(i)

    # State for backtracking
    # assignment: list of -1, 0, 1
    # count_0[i]: number of vertices in edge i assigned 0
    # count_1[i]: number of vertices in edge i assigned 1
    # unassigned[i]: number of vertices in edge i unassigned
    
    assignment = [-1] * n
    count_0 = [0] * len(edges)
    count_1 = [0] * len(edges)
    unassigned = [5] * len(edges) # Since 5-uniform
    
    # We need to track which vertices are unassigned in each edge for unit propagation?
    # Actually, if count_0 == 4, the remaining 1 MUST be 1.
    # If count_1 == 4, the remaining 1 MUST be 0.
    
    # To make it faster, we can just re-scan edges or maintain counts.
    # Given small n (9-12) and small E (<50), re-scanning or simple counts are fine.
    # But we need to find the vertex to assign.
    
    # Heuristic: Pick unassigned vertex with highest degree in remaining constraints?
    # Or just first unassigned.
    
    def backtrack():
        # Unit Propagation Check
        # If any edge has 4 of one color, the 5th is forced.
        # If any edge has 5 of one color, conflict.
        
        # Find next unassigned vertex
        # We can pick the one with the most constrained edges.
        # Simple version: first unassigned.
        next_v = -1
        max_constraints = -1
        for v in range(n):
            if assignment[v] == -1:
                # Count how many edges involving v are "tight" (4 assigned)
                # Or just use degree.
                # Let's use a simple heuristic: highest degree among unassigned.
                if deg[v] > max_constraints:
                    max_constraints = deg[v]
                    next_v = v
                    
        if next_v == -1:
            return True # All assigned, no conflicts found (conflicts checked below)
            
        # Try 0
        assignment[next_v] = 0
        valid = True
        affected_edges = v_to_edges[next_v]
        
        # Update counts for affected edges
        for ei in affected_edges:
            count_0[ei] += 1
            unassigned[ei] -= 1
            if count_0[ei] == 5:
                valid = False
                break
            if count_0[ei] == 4:
                # The remaining vertex in this edge must be 1
                # Find it
                for v in edges[ei]:
                    if assignment[v] == -1:
                        if v != next_v: # next_v is 0, so this is the other one
                            # Force it to 1
                            # We need to recursively assign or just check if it's already 1?
                            # If it's unassigned, we must assign it.
                            # This complicates simple backtracking. 
                            # For n=9, we can just let the recursion handle it, 
                            # but unit propagation prunes branches.
                            pass
        
        if valid:
            # Check for conflicts in all edges? 
            # Or just affected? 
            # If we don't propagate, we might miss a conflict where an edge becomes 5-0 
            # due to a previous assignment? No, we check on assignment.
            # But if we don't propagate, we might assign a vertex that creates a 5-0 in 
            # an edge we didn't check? No, we check affected edges.
            # However, if we don't force the 4->1 case, we might explore a branch 
            # that is dead-end because the 5th vertex is forced.
            
            # Let's do a simpler check: 
            # After assigning next_v, check all edges for monochromatic.
            # If found, backtrack.
            # This is O(E * 5) per step. E~50, 5=250 ops. Depth 9. 2^9 * 250 ~ 100k ops. Very fast.
            
            conflict = False
            for ei in range(len(edges)):
                if count_0[ei] == 5 or count_1[ei] == 5:
                    conflict = True
                    break
            
            if not conflict and backtrack():
                return True
        
        # Undo 0
        for ei in affected_edges:
            count_0[ei] -= 1
            unassigned[ei] += 1
        assignment[next_v] = -1
        
        # Try 1
        assignment[next_v] = 1
        valid = True
        for ei in affected_edges:
            count_1[ei] += 1
            unassigned[ei] -= 1
            if count_1[ei] == 5:
                valid = False
                break
        
        if valid:
            conflict = False
            for ei in range(len(edges)):
                if count_0[ei] == 5 or count_1[ei] == 5:
                    conflict = True
                    break
            
            if not conflict and backtrack():
                return True
        
        # Undo 1
        for ei in affected_edges:
            count_1[ei] -= 1
            unassigned[ei] += 1
        assignment[next_v] = -1
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep margin
    
    best_edges = None
    best_count = 201  # Initialize to max allowed + 1
    
    # We want to minimize edges.
    # Known: n=9, complete graph K_9^5 has 126 edges.
    # We can try to find a small subset.
    # Strategy: Randomized Greedy Removal on small n.
    # Or: Randomized construction.
    
    # Let's try n=9 first, as it's the smallest possible for 5-uniform (n>=5).
    # Actually, n=5 has 1 edge. Is it 2-colorable? Yes (00001 or 11110 etc, wait, 5 vertices. 
    # Edge {0,1,2,3,4}. Coloring 0,0,0,0,1 works. So n=5 is 2-colorable.
    # n=6? Max edges 6. Probably 2-colorable.
    # n=7? 
    # n=8?
    # n=9?
    
    # Let's iterate n from 9 to 12.
    # For each n, try to find a non-2-colorable hypergraph with few edges.
    
    for n in range(9, 13):
        if time.time() > deadline - 10:
            break
            
        all_edges = [tuple(e) for e in itertools.combinations(range(n), 5)]
        
        # Randomize the order of edges to break symmetry
        random.shuffle(all_edges)
        
        # Start with a random subset? 
        # Or start with all and remove?
        # Removal is better if we start with a known non-2-colorable set.
        # The complete hypergraph K_n^5 is non-2-colorable for n >= 9?
        # Actually, the property B threshold is around n=9 for 5-uniform?
        # Let's check if K_9^5 is non-2-colorable.
        
        current_edges = list(all_edges)
        
        # Check if current is non-2-colorable
        if _is_2_colorable_fast(n, current_edges):
            continue # It's 2-colorable, so we can't use this n as a base for removal
        
        # If it's non-2-colorable, try to remove edges.
        # Greedy removal: 
        # Shuffle current_edges to try different orders
        for attempt in range(5): # Try 5 different random orders
            if time.time() > deadline - 5:
                break
                
            random.shuffle(current_edges)
            removed = True
            while removed and time.time() < deadline - 2:
                removed = False
                # Iterate through edges
                for i in range(len(current_edges) - 1, -1, -1):
                    if time.time() > deadline - 2:
                        break
                    
                    candidate = current_edges[:i] + current_edges[i+1:]
                    # Check if candidate is still non-2-colorable
                    if not _is_2_colorable_fast(n, candidate):
                        current_edges = candidate
                        removed = True
                        if len(current_edges) < best_count:
                            best_count = len(current_edges)
                            best_edges = [list(e) for e in current_edges]
                            # Update global best
                            # Note: best_edges must be for this n. 
                            # We need to return n as well.
                            # The return format is {'n': n, 'edges': ...}
                            # So we store (n, edges)
                            pass
        
        # Store best for this n
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            # We need to remember the n associated with best_edges
            # Let's use a variable best_n
            best_n = n

    if best_edges is None:
        # Fallback: return a known small non-2-colorable hypergraph if found, 
        # or just a valid one.
        # If none found, return a dummy? No, must be valid.
        # K_9^5 is valid.
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}

# Helper to track best_n properly in the loop above
# The code above has a slight bug: best_n is only defined inside the if block.
# Let's fix that by initializing best_n = -1.

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep margin
    
    best_edges = None
    best_count = 201  # Initialize to max allowed + 1
    best_n = -1
    
    # We want to minimize edges.
    # Strategy: Randomized Greedy Removal on small n.
    
    for n in range(9, 13):
        if time.time() > deadline - 10:
            break
            
        all_edges = [tuple(e) for e in itertools.combinations(range(n), 5)]
        
        # Randomize the order of edges to break symmetry
        random.shuffle(all_edges)
        
        current_edges = list(all_edges)
        
        # Check if current is non-2-colorable
        if _is_2_colorable_fast(n, current_edges):
            continue # It's 2-colorable, so we can't use this n as a base for removal
        
        # If it's non-2-colorable, try to remove edges.
        # Greedy removal: 
        # Shuffle current_edges to try different orders
        for attempt in range(10): # Try 10 different random orders
            if time.time() > deadline - 5:
                break
                
            random.shuffle(current_edges)
            removed = True
            while removed and time.time() < deadline - 2:
                removed = False
                # Iterate through edges
                for i in range(len(current_edges) - 1, -1, -1):
                    if time.time() > deadline - 2:
                        break
                    
                    candidate = current_edges[:i] + current_edges[i+1:]
                    # Check if candidate is still non-2-colorable
                    if not _is_2_colorable_fast(n, candidate):
                        current_edges = candidate
                        removed = True
                        if len(current_edges) < best_count:
                            best_count = len(current_edges)
                            best_edges = [list(e) for e in current_edges]
                            best_n = n
                            # Note: we update best immediately to allow early exit if we hit 51?
                            # No, we want to minimize.
        
        # Store best for this n (in case the inner loop didn't update best_n for some reason, 
        # though it should if current_edges is smaller than previous best)
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            best_n = n

    if best_edges is None:
        # Fallback: return a known small non-2-colorable hypergraph if found, 
        # or just a valid one.
        # If none found, return K_9^5.
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}
