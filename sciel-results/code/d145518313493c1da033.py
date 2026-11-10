# EVOLVE-BLOCK-START
import itertools
import time

def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    n: number of vertices
    edges: list of sets of vertices
    """
    # Convert edges to sets of indices for faster lookup
    edge_sets = [frozenset(e) for e in edges]
    
    # colors: -1 unknown, 0 color A, 1 color B
    colors = [-1] * n
    
    # Precompute for each vertex, the list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_sets):
        for v in e:
            vertex_edges[v].append(i)
    
    def backtrack(idx=0):
        if idx >= n:
            # Check if any edge is monochromatic
            for e in edge_sets:
                c = colors[list(e)[0]]
                if all(colors[v] == c for v in e):
                    return False
            return True
            
        v = idx
        # Try color 0
        for c in [0, 1]:
            # Check if this assignment is consistent
            valid = True
            for e in vertex_edges[v]:
                # Check if this edge is already forced to be monochromatic of the other color?
                # Or if all other vertices in e are c, then v cannot be c?
                # Actually, if all other vertices in e are c, then v MUST be 1-c to avoid monochromatic?
                # No, if all other vertices are c, and we set v to c, the edge becomes monochromatic c.
                # So we cannot set v to c if all other vertices in e are c.
                # Similarly, if all other vertices are 1-c, we cannot set v to 1-c.
                
                others = [colors[u] for u in e if u != v]
                if all(x == c for x in others if x != -1):
                    # If all known others are c, and we set v to c, edge is mono c.
                    # But wait, if some are -1, we don't know yet.
                    # Unit propagation:
                    # If an edge has 4 vertices of color c and 1 unknown, the unknown MUST be 1-c.
                    # If an edge has 5 vertices of color c, it's invalid.
                    
                    # Let's just do simple consistency check:
                    # If we assign v=c, does any edge become monochromatic c?
                    # That requires all other vertices in e to be c.
                    if all(x == c for x in others):
                        valid = False
                        break
                    # Does any edge become monochromatic 1-c?
                    # That requires v=1-c and all others=1-c.
                    # We are trying v=c, so this is not directly triggered by v=c unless v was 1-c.
                    # But we check if v=c forces a contradiction.
                    # Contradiction arises if an edge has all vertices c.
                    # Or if an edge has 4 vertices 1-c and v is forced to be c? No.
                    # If an edge has 4 vertices 1-c and v is c, it's mixed.
                    # If an edge has 4 vertices c and v is c, it's mono c. (Handled above)
                    
                    # What if an edge has 4 vertices 1-c and v is -1?
                    # Then v MUST be c.
                    # If we are trying v=0, and some edge forces v=1, then v=0 is invalid.
                    # Let's implement unit propagation properly.
                    pass
                    
            if not valid:
                continue
                
            colors[v] = c
            # Apply unit propagation
            if propagate():
                if backtrack(idx + 1):
                    return True
            colors[v] = -1
        return False

    def propagate():
        # Repeat until no change
        changed = True
        while changed:
            changed = False
            for e in edge_sets:
                cs = [colors[v] for v in e]
                if -1 not in cs:
                    if all(x == cs[0] for x in cs):
                        return False # Monochromatic edge found
                else:
                    # Count colors
                    c0 = cs.count(0)
                    c1 = cs.count(1)
                    if c0 == 4:
                        # One unknown must be 1
                        for v in e:
                            if colors[v] == -1:
                                colors[v] = 1
                                changed = True
                    elif c1 == 4:
                        # One unknown must be 0
                        for v in e:
                            if colors[v] == -1:
                                colors[v] = 0
                                changed = True
        return True

    if propagate():
        if backtrack(0):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin
    
    # Start with K9^(5)
    n = 9
    edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Greedy removal
    # We want to remove edges that are "redundant" for non-2-colorability.
    # An edge is redundant if removing it leaves the hypergraph non-2-colorable.
    
    # To speed up, we can randomize the order of edges to try.
    # But we need to be deterministic or seed it.
    import random
    random.seed(42)
    
    # Sort edges? Or just iterate.
    # We will iterate through edges multiple times.
    
    # Optimization: If time is running out, return current best.
    
    # Initial check: K9 is non-2-colorable.
    # We try to remove edges.
    
    # To make it faster, we can use a heuristic:
    # Edges that are "central" might be more important.
    # But let's just try all.
    
    # Since checking 2-colorability is O(2^n) worst case, but n=9 is small (512 states),
    # it is very fast.
    # We can do this.
    
    # We'll keep a list of current edges.
    current_edges = edges[:]
    
    # To allow more search, we can try to remove edges in a specific order.
    # We'll shuffle the edges to avoid bias.
    edge_indices = list(range(len(current_edges)))
    random.shuffle(edge_indices)
    
    # We'll do passes.
    # In each pass, we try to remove the first edge that can be removed.
    # Or we check all and remove all possible at once?
    # Removing one at a time is safer to ensure the check is on the current state.
    
    # Given 126 edges, and each check is fast, we can do many passes.
    
    # Let's try to remove as many as possible.
    
    # We will use a set for faster edge lookup?
    # The _is_2_colorable function takes a list.
    
    # We can optimize _is_2_colorable to take a list of tuples.
    
    # Let's refine the loop.
    
    removed_any = True
    while removed_any and (time.time() < deadline):
        removed_any = False
        # Shuffle to ensure we don't get stuck in a local minimum of removal order?
        # Actually, the order of removal matters for the final set, but any minimal set is valid.
        # We want a small set.
        # Greedy removal usually yields a small set.
        
        # We iterate through the current edges.
        # If we remove one, we restart the loop (or continue with the new list).
        
        # To be efficient, we check edges in random order.
        indices = list(range(len(current_edges)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            # Try removing edge i
            # Create a new list without edge i
            # This is O(E). E is small.
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if not _is_2_colorable(n, test_edges):
                # It is still non-2-colorable, so we can remove it.
                current_edges = test_edges
                removed_any = True
                # Restart the while loop? Or continue?
                # If we continue, we might remove more.
                # But the indices are now invalid.
                # So we break and restart the while loop.
                break
                
    return {'n': n, 'edges': current_edges}

# EVOLVE-BLOCK-END
