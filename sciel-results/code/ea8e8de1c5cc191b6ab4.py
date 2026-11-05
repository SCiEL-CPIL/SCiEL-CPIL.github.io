import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Map vertices to indices 0..n-1
    # Use bitmasks for efficiency? n <= 26, so Python ints are fine.
    
    # Pre-compute edges as lists of vertex indices
    # For unit propagation, we need to know which edges are "monochromatic" 
    # or "almost monochromatic".
    
    # Actually, a simpler backtracking:
    # Assign colors to vertices.
    # Constraint: No edge is monochromatic.
    
    # We can use a SAT-like approach or simple backtracking with pruning.
    
    # Let's use a simple backtracking with bitsets for edges.
    # But n is small (<=26), edges <= 200.
    
    # We'll use a recursive function.
    # To speed up, we can order vertices by degree or just 0..n-1.
    
    # Unit propagation:
    # If an edge has 4 vertices of color 0, the 5th MUST be 1.
    # If an edge has 4 vertices of color 1, the 5th MUST be 0.
    # If an edge has 5 vertices of same color, conflict.
    
    # We'll maintain:
    # color[v] = -1 (unassigned), 0, or 1
    # For each edge, count of 0s and 1s.
    
    # To make it fast, we'll use lists.
    
    # Initialize
    color = [-1] * n
    # edge_counts[i] = [count_0, count_1]
    edge_counts = [[0, 0] for _ in edges]
    
    # For each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    # Pre-sort vertices by degree descending?
    # Or just use fixed order.
    
    # Backtracking function
    def backtrack(v_idx):
        if v_idx == n:
            return True
            
        v = v_idx
        # Try color 0
        # Check if assigning 0 to v causes any edge to become monochromatic 0?
        # Or forces a conflict?
        
        # We need to propagate.
        # This is complex to implement correctly and fast in Python within 120s for many checks.
        
        # Alternative: Use a SAT solver? No, standard library only.
        # Alternative: Simple backtracking without unit propagation might be too slow if the hypergraph is dense.
        # But we start with complete hypergraph.
        
        # Let's try a simpler check:
        # Randomized coloring? No, must be exact.
        
        # Given the small size, maybe a simple backtracking is okay for the *removal* step?
        # We only need to check if it's 2-colorable.
        
        # Let's use a bitset representation for the search space?
        
        # Actually, for the "removal" step, we only need to check if the *remaining* graph is 2-colorable.
        # If it IS 2-colorable, we keep the edge.
        # If it is NOT, we remove it.
        
        # So we need a fast "is_2_colorable" check.
        
        # Let's implement a basic backtracking with some heuristics.
        
        # Heuristic: Pick vertex with most constraints?
        # Or just 0..n-1.
        
        # We'll use a global timeout for the whole construct() function.
        
        pass # Placeholder

    # Since implementing a fast exact 2-colorability checker is tricky, 
    # and the current program returns a valid (but large) hypergraph,
    # maybe we should just return a known smaller one if we can hardcode it?
    
    # But I don't have a 51-edge one hardcoded.
    
    # Let's try to implement the checker and the removal loop.
    
    # Optimized Checker:
    # Use bitmasks for edges?
    
    # Let's try a different approach: 
    # The complete hypergraph on 9 vertices is non-2-colorable.
    # We can try to remove edges randomly and check.
    
    # But the check is the bottleneck.
    
    # Let's write a reasonably fast backtracking checker.
    
    def check_2c(n, edges):
        # Use bitsets for vertex colors?
        # color_mask_0, color_mask_1
        # Unassigned: ~ (color_mask_0 | color_mask_1)
        
        # For each edge, we need to know if it's monochromatic.
        
        # We'll use recursion.
        
        # To prune:
        # If an edge has 4 vertices in color 0, the 5th must be 1.
        
        # This is unit propagation.
        
        # We'll implement a queue for unit propagation.
        
        # State:
        # color: list of -1, 0, 1
        # edge_status: for each edge, count of 0s and 1s.
        
        # This is too slow in Python if done naively.
        
        # Let's assume the check takes ~1ms. 126 edges * 126 removals = 15876 checks. 15s. Feasible.
        
        # Implementation of check_2c:
        
        # We'll use a simple backtracking with MRV (Minimum Remaining Values) heuristic?
        # Or just fixed order.
        
        # Let's try fixed order 0..n-1.
        
        color = [-1] * n
        edge_counts = [[0, 0] for _ in edges]
        
        # Precompute vertex to edges
        v_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_edges[v].append(i)
                
        # To speed up, we can precompute the edges as tuples for faster hashing?
        
        def propagate():
            # Check for conflicts and force assignments
            # This is a simplified version.
            # We'll just check if any edge is monochromatic.
            # And if any edge has 4 of one color, force the 5th.
            
            changed = True
            while changed:
                changed = False
                for i, e in enumerate(edges):
                    c0, c1 = edge_counts[i]
                    if c0 == 5 or c1 == 5:
                        return False # Conflict
                    if c0 == 4:
                        # The unassigned vertex in this edge must be 1
                        for v in e:
                            if color[v] == -1:
                                # Force v to 1
                                color[v] = 1
                                edge_counts[i][1] += 1
                                # Update other edges containing v
                                for ei in v_edges[v]:
                                    if ei != i:
                                        edge_counts[ei][1] += 1
                                changed = True
                                break
                    elif c1 == 4:
                        for v in e:
                            if color[v] == -1:
                                color[v] = 0
                                edge_counts[i][0] += 1
                                for ei in v_edges[v]:
                                    if ei != i:
                                        edge_counts[ei][0] += 1
                                changed = True
                                break
            return True

        def backtrack(v_idx):
            if v_idx == n:
                return True
            
            v = v_idx
            # Try 0
            # Save state? 
            # For simplicity, we'll just recurse. 
            # But we need to restore state.
            # This is getting complicated.
            
            # Let's use a simpler check: 
            # Is the hypergraph 2-colorable?
            # We can use a SAT encoding and a simple DPLL?
            
            # Given the complexity, let's just return the current 126-edge one if we can't implement the search quickly.
            
            # BUT, the prompt asks to rewrite with a stronger algorithm.
            
            # Let's try to find a known small 5-uniform non-2-colorable hypergraph.
            # The "best known" is 51.
            
            # Maybe we can use a random construction and verify?
            
            # Let's try to generate a random 5-uniform hypergraph on 9 vertices with ~50 edges and check if it's non-2-colorable.
            # If yes, return it.
            
            # But we need to be sure it's non-2-colorable.
            
            # Let's implement a fast checker using bitsets and backtracking.
            
            pass

    # Let's try to implement the checker properly.
    
    # We'll use a bitset for the color assignment.
    # But Python doesn't have native bitsets, we use ints.
    
    # Let's try a different strategy:
    # Return a hypergraph that is known to be non-2-colorable and has fewer edges.
    
    # Actually, I can construct one:
    # Take the 5-uniform hypergraph on 9 vertices.
    # Remove all edges that contain vertex 0? 
    # Then the remaining vertices are 1..8.
    # If the subgraph on 1..8 is 2-colorable, then the whole graph is 2-colorable (color 0 as 0, rest as per subgraph).
    # So we need the subgraph on 1..8 to be non-2-colorable.
    # The complete 5-uniform hypergraph on 8 vertices has C(8,5)=56 edges.
    # Is it non-2-colorable?
    # Any 2-coloring of 8 vertices has 4 of one color.
    # If we have 5 vertices of one color, we have a monochromatic edge.
    # In a 2-coloring of 8 vertices, the max size of a color class is 4? No, it can be 8-0, 7-1, etc.
    # Wait, "proper 2-coloring" means NO edge is monochromatic.
    # So we need to avoid any 5-subset being all 0 or all 1.
    # If we color 4 vertices 0 and 4 vertices 1, there is no monochromatic 5-subset.
    # So the complete 5-uniform hypergraph on 8 vertices IS 2-colorable (4-4 split).
    
    # So we need more vertices or a specific structure.
    
    # The complete 5-uniform hypergraph on 9 vertices:
    # If we color 5 vertices 0 and 4 vertices 1, the 5 zeros form a monochromatic edge.
    # If we color 4 vertices 0 and 5 vertices 1, the 5 ones form a monochromatic edge.
    # Any 2-coloring of 9 vertices has at least 5 of one color.
    # So it is NOT 2-colorable.
    
    # Can we remove edges from the 9-vertex complete one and still be non-2-colorable?
    # Yes.
    
    # Let's try to remove edges that are "redundant".
    
    # I will implement the removal loop with a fast checker.
    
    pass

def construct():
    # Implementation of the removal strategy
    import time
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    edges = all_edges[:]
    
    # Fast 2-colorability checker
    def is_2_colorable(n, edges):
        # Use backtracking with unit propagation
        # Represent colors as list of -1, 0, 1
        # This is a simplified version for speed
        
        if not edges:
            return True
            
        # Precompute
        v_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_edges[v].append(i)
                
        # We'll use a recursive function with bitsets?
        # Let's try a simple heuristic: 
        # If the hypergraph is dense, it's likely non-2-colorable.
        # But we need exact check.
        
        # Let's use a SAT-like DPLL.
        
        # Actually, for n=9, the search space is 2^9 = 512 colorings.
        # We can just iterate through all 512 colorings and check if any is valid!
        
        # A coloring is valid if no edge is monochromatic.
        
        # For each coloring (mask from 0 to 511):
        # Check all edges.
        
        # This is 512 * len(edges) operations.
        # len(edges) <= 126.
        # 512 * 126 = 64512 operations. Very fast.
        
        for mask in range(1 << n):
            # Check if this coloring is valid
            valid = True
            for e in edges:
                # Check if all vertices in e have same color
                # Get color of first vertex
                c0 = (mask >> e[0]) & 1
                mono = True
                for v in e[1:]:
                    cv = (mask >> v) & 1
                    if cv != c0:
                        mono = False
                        break
                if mono:
                    valid = False
                    break
            if valid:
                return True
                
        return False

    # Removal loop
    # We want to remove edges while keeping it non-2-colorable.
    # So if removing an edge makes it 2-colorable, we don't remove it.
    
    # Order of removal: random or fixed?
    # Fixed order might be biased. Random might find smaller ones.
    # Let's try random order.
    
    import random
    random.seed(42)
    
    # We'll try to remove edges one by one.
    # To avoid getting stuck in a local minimum, we can restart if no progress.
    
    # But for now, let's just do a single pass.
    
    # Actually, the order matters.
    # Let's try to remove edges in a random order.
    
    indices = list(range(len(edges)))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline:
            break
            
        # Try to remove edge at idx
        if idx >= len(edges):
            continue
            
        e_to_remove = edges[idx]
        
        # Remove it
        edges.pop(idx)
        
        # Check if still non-2-colorable
        if not is_2_colorable(n, edges):
            # It is still non-2-colorable, so we keep it removed
            pass
        else:
            # It became 2-colorable, so we restore it
            edges.insert(idx, e_to_remove)
            
    return {'n': n, 'edges': edges}

# Run a quick test to see how many edges we get
# print(construct()['edges'])
