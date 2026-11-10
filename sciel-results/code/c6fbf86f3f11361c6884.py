"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable_fast(n, edges):
    """
    Check if hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Precompute edge bitmasks for fast checking
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # We need to check if there exists a coloring such that no edge is monochromatic.
    # This is equivalent to: for every edge, not(all 0) and not(all 1).
    # We can use a simple backtracking with bitmasks.
    
    # A more efficient approach: 
    # If an edge has 4 vertices of the same color, the 5th must be different (unit propagation).
    # We'll implement a recursive backtracker with bitsets.
    
    # Let's use a different perspective: 
    # A hypergraph is NOT 2-colorable iff it contains a "critical" substructure.
    # But checking non-colorability is hard. Checking colorability is SAT.
    
    # Let's implement a fast SAT solver for this specific structure.
    # Variables: x_0, ..., x_{n-1}
    # Clauses: For each edge {v1..v5}, we add clauses:
    #   (x_v1 OR x_v2 OR x_v3 OR x_v4 OR x_v5)  -> ensures not all 0
    #   (!x_v1 OR !x_v2 OR !x_v3 OR !x_v4 OR !x_v5) -> ensures not all 1
    
    # This is a 5-SAT instance.
    # We can use DPLL with unit propagation.
    
    # State:
    # assignment: -1 (unassigned), 0 (False), 1 (True)
    # clauses: list of sets of (var, polarity)
    
    # For performance, we'll use bitmasks for clauses.
    # Clause 1 of edge i: OR of all 5 vars. 
    # Clause 2 of edge i: OR of all 5 neg vars.
    
    # Let's represent the problem as:
    # We need to find x such that for all edges, sum(x_v) is in {1,2,3,4}.
    
    # Backtracking search:
    # Order variables by degree or just index.
    
    def backtrack(idx, assignments, edge_satisfied, edge_violated):
        # idx: next variable to assign
        # assignments: list of 0/1/-1
        # edge_satisfied: bitmask of edges that are already "safe" (can't become monochromatic)
        # edge_violated: bitmask of edges that are already violated (monochromatic)
        # If edge_violated != 0, return False.
        
        if edge_violated:
            return False
            
        if idx == n:
            # Check if any edge is violated. 
            # Actually, if we maintain edge_violated, and it's 0, and we've assigned all vars,
            # we need to ensure no edge is monochromatic.
            # Wait, "satisfied" logic is tricky.
            # Let's just check at the end if edge_violated is 0.
            # But we also need to ensure no edge is "pending" in a way that it *is* monochromatic.
            # If an edge has 5 assigned vars, and they are all 0 or all 1, it's violated.
            # Our propagation should catch this.
            return edge_violated == 0
        
        # Heuristic: choose variable with most constraints? Or just idx.
        # For simplicity and speed in small n, let's stick to idx or a simple heuristic.
        
        # Try 0
        # Try 1
        
        # Unit Propagation:
        # Before branching, check if any edge is forced.
        # An edge is forced if 4 vars are assigned to 0 and 1 is unassigned -> that 1 must be 1.
        # Or 4 vars are 1 and 1 is unassigned -> that 1 must be 0.
        
        # Let's implement a helper to propagate.
        # This is getting complex to implement efficiently in pure Python within the time limit for many calls.
        
        # Alternative: Use the known 51-edge construction directly and verify it once.
        # Then use local search to try to reduce edges.
        # The verification only needs to happen for the final candidate or during local search steps.
        
        pass
        
    # The recursive function above is a sketch. Let's implement a concrete, fast checker.
    
    # Fast Checker using bitmasks and recursion.
    # We will try to find a coloring.
    
    # Precompute for each edge:
    # mask: 5-bit integer representing the vertices? No, n can be up to 26.
    # Use Python integers as bitsets for vertices.
    
    def check_colorable(edges):
        if not edges:
            return True
            
        # Convert edges to bitmasks
        emasks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            emasks.append(m)
            
        # We use a recursive function.
        # To speed it up, we can use a heuristic: pick a variable that appears in many edges.
        
        # Let's try a simpler approach first: Randomized Local Search for coloring.
        # If we can find a coloring, it's colorable.
        # If we fail after many tries, it's likely not colorable (but not proof).
        # The task requires EXACT check.
        
        # So we MUST use exact backtracking.
        
        # Optimized Backtracking:
        # 1. Order variables by degree (number of edges they belong to).
        # 2. Use forward checking: if an edge becomes monochromatic, backtrack.
        
        # Precompute incidence
        incidence = {v: [] for v in range(n)}
        for i, e in enumerate(edges):
            for v in e:
                incidence[v].append(i)
                
        # Order variables
        var_order = sorted(range(n), key=lambda v: len(incidence[v]), reverse=True)
        var_pos = {v: i for i, v in enumerate(var_order)}
        
        # Recursive search
        # state: assignments (dict or list), 
        #        edge_status: for each edge, count of 0s and 1s.
        #        If count_0 == 5 or count_1 == 5, fail.
        
        assignments = [-1] * n
        edge_0s = [0] * len(edges)
        edge_1s = [0] * len(edges)
        
        def recurse(k):
            if k == n:
                return True
                
            v = var_order[k]
            
            # Try 0
            assignments[v] = 0
            affected_edges = incidence[v]
            valid = True
            for ei in affected_edges:
                edge_0s[ei] += 1
                if edge_0s[ei] == 5:
                    valid = False
                    break
            if valid:
                if recurse(k + 1):
                    return True
            # Undo
            for ei in affected_edges:
                edge_0s[ei] -= 1
                
            # Try 1
            assignments[v] = 1
            valid = True
            for ei in affected_edges:
                edge_1s[ei] += 1
                if edge_1s[ei] == 5:
                    valid = False
                    break
            if valid:
                if recurse(k + 1):
                    return True
            # Undo
            for ei in affected_edges:
                edge_1s[ei] -= 1
                
            assignments[v] = -1
            return False
            
        return recurse(0)
        
    return check_colorable(edges)

def construct():
    deadline = time.time() + 145
    random.seed(42)
    
    # Known 51-edge construction for 5-uniform hypergraph on 9 vertices?
    # Actually, the complete hypergraph on 9 vertices has C(9,5) = 126 edges.
    # The task says best known is 51.
    # Let's start with the complete hypergraph on 9 vertices and try to remove edges.
    # Or better, start with a random set of 51 edges and verify, then minimize.
    
    # Let's use the "greedy removal" strategy but with a fast checker.
    # Start with all 126 edges on 9 vertices.
    n = 9
    all_edges = list(combinations(range(n), 5))
    
    # Check if complete hypergraph is not colorable?
    # C(9,5) is large. It is likely not colorable.
    # Let's verify.
    if _is_2_colorable_fast(n, all_edges):
        # If it is colorable, we need a different starting point.
        # But usually complete hypergraphs are not colorable for small n.
        # For n=9, is it colorable?
        # If we color 4 red, 5 blue. Any edge with 5 blue is bad.
        # If we color 5 red, 4 blue. Any edge with 5 red is bad.
        # So we must avoid having 5 of any color in any edge.
        # This means the red set and blue set must have no 5-subsets.
        # So |Red| < 5 and |Blue| < 5.
        # But |Red| + |Blue| = 9.
        # Max |Red| = 4, Max |Blue| = 4. Sum = 8. Contradiction.
        # So it is NOT colorable.
        pass
    else:
        # It is not colorable. Good.
        pass
        
    current_edges = all_edges[:]
    
    # Greedy removal
    # Shuffle edges to break symmetry
    random.shuffle(current_edges)
    
    removed_count = 0
    for i in range(len(current_edges)):
        if time.time() > deadline - 10:
            break
            
        # Try to remove current_edges[i]
        # To do this efficiently, we can just check if the remaining set is colorable.
        # If it IS colorable, we cannot remove it.
        # If it IS NOT colorable, we can remove it.
        
        # Note: current_edges is a list of tuples.
        # We need to test the set without current_edges[i].
        
        # Optimization: Instead of copying the list, we can pass a flag or use a set.
        # But _is_2_colorable_fast takes a list.
        
        # Let's try removal.
        # We iterate through the list.
        # If we remove an edge, the list shrinks.
        
        # We'll use a while loop with index i.
        pass
        
    # Let's implement the removal loop properly.
    i = 0
    while i < len(current_edges) and time.time() < deadline - 10:
        # Test if removing current_edges[i] makes it colorable.
        test_edges = current_edges[:i] + current_edges[i+1:]
        
        if _is_2_colorable_fast(n, test_edges):
            # It became colorable, so we MUST keep this edge.
            i += 1
        else:
            # It is still not colorable, so we CAN remove this edge.
            current_edges.pop(i)
            removed_count += 1
            # Do not increment i, as the next edge shifted into position i.
            
    return {'n': n, 'edges': [list(e) for e in current_edges]}
# EVOLVE-BLOCK-END
