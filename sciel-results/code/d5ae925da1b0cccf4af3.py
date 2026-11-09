"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _is_3_chromatic(edges, n=9):
    """Check if the hypergraph is NOT 2-colorable (i.e., chromatic number >= 3).
    Since we only use 3 colors and the structure is small, if it's not 2-colorable,
    it must be 3-chromatic (assuming it's 3-colorable, which we can verify or assume
    for the minimal case, but strictly we need to ensure it's not 2-colorable.
    The task says 'exactly three-chromatic', which implies chi=3.
    If a hypergraph is not 2-colorable, its chromatic number is at least 3.
    We must also ensure it IS 3-colorable. However, for small n and r=3,
    most non-2-colorable graphs are 3-colorable. We will verify 3-colorability
    just in case, or rely on the fact that we are searching for the minimal one.
    Actually, the problem states 'Minimize edge count among exactly three-chromatic triple systems'.
    So we need chi(H) = 3.
    This means:
    1. H is NOT 2-colorable.
    2. H IS 3-colorable.
    """
    # 1. Check NOT 2-colorable
    # A hypergraph is 2-colorable if there is a partition of vertices into two sets
    # such that no edge is contained entirely in one set.
    # Equivalently, no 2-coloring of vertices makes any edge monochromatic.
    # We check if there EXISTS a 2-coloring where NO edge is monochromatic.
    # If such a coloring exists, it is 2-colorable.
    # We want to know if it is NOT 2-colorable.
    
    # Efficient check for 2-colorability of hypergraph:
    # This is equivalent to checking if the dual hypergraph is bipartite? No.
    # It's equivalent to checking if the constraint satisfaction problem (CSP) has a solution.
    # Variables: vertices (9). Domain: {0, 1}.
    # Constraints: For each edge {u,v,w}, not (c[u]==c[v]==c[w]).
    # This is a SAT problem. With 9 variables, 2^9 = 512 assignments.
    # We can iterate through all 512 assignments and check if any satisfies all constraints.
    
    all_two_colorings = itertools.product([0, 1], repeat=n)
    for coloring in all_two_colorings:
        is_valid = True
        for e in edges:
            if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                is_valid = False
                break
        if is_valid:
            # Found a valid 2-coloring, so it IS 2-colorable.
            return False
    
    # If we reach here, it is NOT 2-colorable.
    # Now check if it IS 3-colorable.
    # Variables: vertices (9). Domain: {0, 1, 2}.
    # Constraints: For each edge {u,v,w}, not (c[u]==c[v]==c[w]).
    # 3^9 = 19683 assignments.
    all_three_colorings = itertools.product([0, 1, 2], repeat=n)
    for coloring in all_three_colorings:
        is_valid = True
        for e in edges:
            if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                is_valid = False
                break
        if is_valid:
            # Found a valid 3-coloring, so it IS 3-colorable.
            return True
    
    # If not 3-colorable, it's not exactly 3-chromatic (it's >3).
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 seconds margin
    
    n = 9
    all_vertices = list(range(n))
    all_triples = list(itertools.combinations(all_vertices, 3))
    
    # We want to find the minimum number of edges.
    # Let's try k = 1, 2, 3, ...
    # For each k, iterate through all combinations of k edges from all_triples.
    # Check if the hypergraph is exactly 3-chromatic.
    # If yes, return it.
    
    # Optimization: 
    # The minimum number of edges for a 3-chromatic hypergraph on n vertices with rank r
    # is known to be related to the Turan number or similar.
    # For n=9, r=3, the minimal 3-chromatic hypergraph is likely small.
    # Let's start from k=1 and go up.
    
    # Note: A single edge is always 2-colorable (and 3-colorable).
    # Two edges: If they share vertices, can we 2-color?
    # Example: {0,1,2}, {2,3,4}. 
    # 0,1,2 must not be monochromatic. 2,3,4 must not be monochromatic.
    # We can always 2-color this.
    
    # Let's just search.
    # To speed up, we can prune.
    # If a set of edges is 2-colorable, any superset is not necessarily 2-colorable?
    # No, adding edges makes it HARDER to 2-color.
    # If a set E is 2-colorable, a superset E' might not be.
    # If a set E is NOT 2-colorable, any superset E' is also NOT 2-colorable.
    # This is a monotonic property for "NOT 2-colorable".
    # However, we also need "IS 3-colorable". Adding edges makes it harder to 3-color too.
    
    # Strategy:
    # Iterate k from 1 to ...
    # For each combination of k edges:
    #   1. Check if NOT 2-colorable. If yes, check if IS 3-colorable.
    #   2. If both, return.
    
    # To avoid iterating millions of combinations if k is large, we need a good bound.
    # For n=9, total triples = C(9,3) = 84.
    # k=1: C(84,1) = 84
    # k=2: C(84,2) = 3486
    # k=3: C(84,3) = 95284
    # k=4: C(84,4) = 2268840 -> ~2.2M. This might be too slow in Python if done naively.
    # But most will be 2-colorable.
    
    # We can optimize the check for 2-colorability.
    # Instead of iterating all 512 colorings, we can use a backtracking solver or
    # simply note that for small k, it's fast.
    
    # Let's try to find a known minimal example first or use a smarter search.
    # Actually, let's just run the search with a time check.
    
    # We can also use the fact that if we find a set of edges that is NOT 2-colorable,
    # we just need to check 3-colorability once.
    
    # To speed up the 2-colorability check:
    # We can represent the constraints.
    # But 512 checks per hypergraph is fine if the number of hypergraphs is small.
    
    # Let's try k=1 to k=6.
    
    for k in range(1, 10):
        if time.time() > deadline:
            break
            
        # Generate combinations of k edges
        # For large k, this is huge. We might need to stop early.
        # But the problem asks to minimize edge count. The answer is likely small.
        
        # If k is large, C(84, k) is huge.
        # We should probably limit k or use a random search if exhaustive is too slow.
        # But let's try exhaustive first for small k.
        
        if k > 5:
            # If we haven't found it by k=5, exhaustive search is too slow.
            # We can switch to a random search or break and return a known good one if we have one.
            # For now, let's just break and return a hardcoded one if we reach here?
            # No, we must return a valid one.
            # Let's try to be smart.
            pass
        
        # To avoid memory issues with itertools.combinations for large k,
        # we can iterate directly.
        
        for edge_set in itertools.combinations(all_triples, k):
            if time.time() > deadline:
                return _fallback()
                
            if _is_3_chromatic(edge_set, n):
                # Found it!
                # We need to provide a coloring.
                # _is_3_chromatic already found one, but we didn't store it.
                # Let's modify _is_3_chromatic to return the coloring or find it again.
                # Finding it again is fast (19683 checks max).
                
                # Find a valid 3-coloring
                all_three_colorings = itertools.product([0, 1, 2], repeat=n)
                valid_coloring = None
                for coloring in all_three_colorings:
                    is_valid = True
                    for e in edge_set:
                        if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                            is_valid = False
                            break
                    if is_valid:
                        valid_coloring = coloring
                        break
                
                if valid_coloring:
                    return {
                        'edges': [list(e) for e in edge_set],
                        'colors': list(valid_coloring)
                    }
                
                # Should not happen if _is_3_chromatic returned True
                break
        
        if time.time() > deadline:
            break

    # If we didn't find one in time, use a known small 3-chromatic hypergraph.
    # One example: The Fano plane is 7 vertices, 7 edges. It is 3-chromatic?
    # Fano plane is 2-colorable? No, it's not.
    # Wait, the Fano plane has chromatic number 3?
    # Actually, the Fano plane is not 2-colorable. Is it 3-colorable? Yes.
    # So Fano plane (n=7) is 3-chromatic.
    # We need n=9. We can add 2 isolated vertices to the Fano plane.
    # Edges: 7. Vertices: 7 used, 2 isolated.
    # This is a valid 3-uniform hypergraph on 9 vertices with 7 edges.
    # Is 7 the minimum?
    # The current seed program uses C(5,3)=10 edges.
    # So 7 is better than 10.
    # Can we do better than 7?
    # The minimum number of edges in a 3-chromatic 3-uniform hypergraph is known.
    # For n=9, it might be smaller.
    # But 7 is a safe bet if the search fails.
    
    return _fallback()

def _fallback():
    # Fano plane on vertices 0..6, plus vertices 7,8 isolated.
    # Fano plane edges:
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    # Check if this is 3-chromatic.
    # It is known to be 3-chromatic.
    # We need a valid 3-coloring.
    # Let's find one.
    n = 9
    edges = fano_edges
    all_three_colorings = itertools.product([0, 1, 2], repeat=n)
    for coloring in all_three_colorings:
        is_valid = True
        for e in edges:
            if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                is_valid = False
                break
        if is_valid:
            return {
                'edges': edges,
                'colors': list(coloring)
            }
    
    # Should not reach here
    return {
        'edges': [[0, 1, 2]],
        'colors': [0, 1, 2, 0, 0, 0, 0, 0, 0]
    }
# EVOLVE-BLOCK-END
