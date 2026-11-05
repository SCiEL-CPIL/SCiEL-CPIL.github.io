"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills after 150s, keep safety margin
_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_time():
    return _elapsed() > _DEADLINE

def _is_2_colorable_sat(n, edges):
    """
    Check if a 5-uniform hypergraph has a proper 2-coloring
    using a 2-SAT reduction.
    
    Variables: x_v is True if vertex v is color 0, False if color 1.
    Constraint: No edge is monochromatic.
    For an edge e = {v1, ..., v5}:
    - Not all 0: (v1 OR v2 OR v3 OR v4 OR v5)  -> This is NOT a 2-SAT clause directly.
    
    Wait, the 2-coloring of hypergraphs is generally NP-hard. 
    However, the prompt says "checked exactly by backtracking with unit propagation".
    The current seed uses vertex backtracking.
    
    The inspiration suggests: "Use a 2-SAT reduction to verify non-2-colorability".
    Is 5-uniform hypergraph 2-colorability reducible to 2-SAT?
    No, k-uniform hypergraph 2-colorability is NP-complete for k>=3.
    BUT, the prompt says "checked exactly by backtracking with unit propagation".
    This implies the checker is a specific algorithm, not just any SAT solver.
    The "2-SAT reduction" inspiration might be misleading or referring to a specific sub-structure 
    OR it implies that we should treat the *search for a coloring* as a CSP and use 
    efficient propagation.
    
    Actually, let's look at the constraint again.
    "No edge is monochromatic".
    If we fix 4 vertices of an edge to 0, the 5th MUST be 1.
    This is unit propagation.
    
    The "2-SAT" idea might be a red herring or a specific trick for *some* structures, 
    but for general 5-uniform, it's not 2-SAT.
    HOWEVER, the vertex backtracking in the seed is O(2^n) in worst case, but with propagation it's often much faster.
    
    Let's stick to a highly optimized backtracking with strong propagation, 
    but change the *construction* strategy.
    
    Strategy: 
    1. Start with a known small non-2-colorable 5-uniform hypergraph or generate one.
    2. Use a "Minimal Unsatisfiable Core" approach: 
       - Start with a graph that is definitely not 2-colorable (e.g., complete graph on 9 vertices has 126 edges, definitely not 2-colorable? 
         Actually, complete 5-uniform on 9 vertices: 
         Any 2-coloring of 9 vertices has at least 5 of one color (pigeonhole). 
         Those 5 form an edge. So K_9^5 is not 2-colorable.
       - Greedily remove edges. If the graph becomes 2-colorable, put the edge back.
       - To speed this up, we need a very fast 2-colorability checker.
    
    Let's implement a faster checker using bitmasks and recursive propagation.
    """
    # Preprocess edges for fast access
    # For each vertex, list of edges containing it
    n_edges = len(edges)
    if n_edges == 0:
        return True
    
    # We will use a recursive backtracking with unit propagation.
    # To make it fast, we represent colors as integers: -1 (unassigned), 0, 1.
    # We maintain for each edge: count0, count1, unassigned_list.
    
    # Optimization: 
    # 1. Sort vertices by degree (most constrained first) for the branching variable?
    #    Or just iterate 0..n-1? 
    #    Given n <= 26, 0..n-1 is fine, but degree ordering helps propagation.
    
    # Let's use a simpler, robust backtracking.
    colors = [-1] * n
    edge_state = []
    for e in edges:
        c0 = 0
        c1 = 0
        unassigned = []
        for v in e:
            if colors[v] == 0: c0 += 1
            elif colors[v] == 1: c1 += 1
            else: unassigned.append(v)
        edge_state.append([c0, c1, unassigned])
        
    def propagate():
        changed = True
        while changed:
            changed = False
            for i, (c0, c1, unassigned) in enumerate(edge_state):
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 1
                        changed = True
                        # Update edge states
                        for j, (ec0, ec1, eun) in enumerate(edge_state):
                            if v in eun:
                                # Remove v from unassigned
                                idx = eun.index(v)
                                eun.pop(idx)
                                # Update counts
                                if j == i:
                                    continue # Already handled? No, need to update i too if v was in i
                                # Actually, simpler to just re-calculate or update carefully.
                                # Let's just update the specific edge j
                                edge_state[j][2] = eun
                                # We don't know if v was 0 or 1 in the count yet, but we know it's now assigned.
                                # This incremental update is tricky. 
                                # Given n<=26 and edges<=200, full recalculation or simple loop is safer.
                                pass
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 0
                        changed = True
                        pass
        # Re-calculate states if changed to ensure consistency? 
        # Or better: just use the simple loop in the seed but optimized with local vars.
        return True

    # The incremental update above is messy. Let's use the simple robust method from the seed 
    # but with a twist: **Randomized Vertex Ordering** and **Conflict-Driven** choice?
    # Or just a very tight loop.
    
    # Let's try a different angle for the checker: 
    # Since n is small (<=26), we can use bitmask representation for colors?
    # No, 2^26 is too big.
    
    # Let's stick to the seed's logic but optimize the inner loop.
    # The seed's logic:
    # propagate():
    #   loop edges:
    #     if 4 of one color, force 5th.
    # backtrack(idx):
    #   for c in (0,1):
    #     colors[idx] = c
    #     if propagate():
    #        if backtrack(idx+1): return True
    #     restore
    
    # This is O(2^n * E * n). With n=26, 2^26 is ~67M. Too slow if not pruned.
    # Propagation is key.
    
    # New Strategy for Construction:
    # 1. Start with K_9^5 (126 edges).
    # 2. Use a "Minimum Edge Removal" heuristic.
    #    Instead of checking removal of edge i one by one (which is O(E) checks per iteration),
    #    we can try to find a minimal unsatisfiable subset (MUS) using a SAT solver approach?
    #    No, we don't have a SAT solver.
    
    # Let's use a **Randomized Greedy Removal** with **Batch Checks**.
    # 1. Start with a non-2-colorable graph.
    # 2. Shuffle edges.
    # 3. Try to remove the first edge. If still non-2-colorable, keep removed. Else, keep edge.
    # 4. Repeat.
    
    # To make the checker faster, we will implement a specialized backtracking 
    # that picks the **most constrained unassigned vertex** (MRV heuristic).
    
    def _check_2_colorable_fast(n, edges):
        if not edges:
            return True
        
        # Build adjacency: for each vertex, which edges contain it
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
                
        colors = [-1] * n
        # edge_counts: [count0, count1] for each edge
        edge_counts = [[0, 0] for _ in range(len(edges))]
        
        def get_unassigned_in_edge(e_idx):
            e = edges[e_idx]
            unassigned = []
            c0 = 0
            c1 = 0
            for v in e:
                if colors[v] == 0: c0 += 1
                elif colors[v] == 1: c1 += 1
                else: unassigned.append(v)
            return c0, c1, unassigned

        def propagate():
            changed = True
            while changed:
                changed = False
                for i in range(len(edges)):
                    c0, c1, unassigned = get_unassigned_in_edge(i)
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and len(unassigned) == 1:
                        v = unassigned[0]
                        if colors[v] == -1:
                            colors[v] = 1
                            changed = True
                        else:
                            # Already assigned to 0? Contradiction
                            if colors[v] == 0:
                                return False
                    elif c1 == 4 and len(unassigned) == 1:
                        v = unassigned[0]
                        if colors[v] == -1:
                            colors[v] = 0
                            changed = True
                        else:
                            if colors[v] == 1:
                                return False
            return True

        def backtrack():
            # Find unassigned vertex with highest degree (most edges)
            # Or just first unassigned? MRV is better.
            min_unassigned_count = len(edges) * 5 # Large
            best_v = -1
            best_degree = -1
            
            for v in range(n):
                if colors[v] == -1:
                    # Degree is number of edges containing v
                    deg = len(vertex_edges[v])
                    # Tie-breaker: random or fixed? Fixed for determinism in checker, 
                    # but random in construction loop? 
                    # Let's use highest degree.
                    if deg > best_degree:
                        best_degree = deg
                        best_v = v
            
            if best_v == -1:
                # All assigned. Check if any edge is monochromatic (should be caught by propagate, 
                # but just in case)
                for i in range(len(edges)):
                    c0, c1, _ = get_unassigned_in_edge(i)
                    if c0 == 5 or c1 == 5:
                        return False
                return True
            
            # Try both colors for best_v
            # Heuristic: Try the color that appears less frequently in its edges?
            # Or just 0 then 1.
            
            # Try 0
            colors[best_v] = 0
            if propagate():
                if backtrack():
                    return True
            colors[best_v] = -1
            
            # Try 1
            colors[best_v] = 1
            if propagate():
                if backtrack():
                    return True
            colors[best_v] = -1
            
            return False

        # Initial propagation
        if not propagate():
            return False
        
        return backtrack()

    return _check_2_colorable_fast(n, edges)

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    """
    # Start with K_9^5
    n = 9
    current_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Verify it's not 2-colorable (it shouldn't be)
    if _is_2_colorable_sat(n, current_edges):
        # This should not happen, but just in case, we need a valid start.
        # If it is 2-colorable, we need to add edges? No, K_9^5 is definitely not.
        pass

    # Greedy removal with randomization
    # To avoid getting stuck in local minima, we can do multiple restarts 
    # or shuffle the order of edge removal attempts.
    
    best_edges = current_edges
    best_count = len(current_edges)
    
    # Time budget for search
    # We have ~140s.
    
    # Strategy: 
    # 1. Try to remove edges one by one in a random order.
    # 2. If removal fails, try the next one.
    # 3. If we remove an edge, restart the scan (or continue).
    # 4. To escape local minima, we can occasionally add a random edge and try to remove others?
    #    No, adding edges makes it harder to be 2-colorable, so it's always safe to add, 
    #    but we want to minimize.
    
    # Better: 
    # Use a "Minimal Subgraph" search.
    # While there exists an edge whose removal preserves non-2-colorability:
    #    Remove it.
    
    # The bottleneck is the check.
    # Let's optimize the loop.
    
    improved = True
    while improved and not _check_time():
        improved = False
        # Shuffle edges to avoid bias
        indices = list(range(len(current_edges)))
        random.shuffle(indices)
        
        for i in indices:
            if _check_time():
                break
            # Try removing edge at index i
            candidate = current_edges[:i] + current_edges[i+1:]
            if not _is_2_colorable_sat(n, candidate):
                current_edges = candidate
                improved = True
                # Reset loop to re-shuffle and try again? 
                # Or just continue? 
                # Continuing might miss interactions. 
                # Restarting the while loop is safer for global optimality but slower.
                # Let's break and restart the while loop.
                break
    
    return {'n': n, 'edges': current_edges}

# Keep the original function name if required by the prompt? 
# The prompt says "Keep the function names and signatures of the seed program."
# Seed had: _is_2_colorable, construct
# I defined _is_2_colorable_sat. I should probably alias it or rename.
# The prompt says "Keep the function names... EXACTLY".
# So I must define _is_2_colorable.

def _is_2_colorable(n, edges):
    return _is_2_colorable_sat(n, edges)

# Final check: The prompt asks for `construct()` -> dict.
# The seed had `_is_2_colorable` and `construct`.
# I will ensure both exist.
