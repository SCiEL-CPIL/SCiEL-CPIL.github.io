import time

def construct():
    """
    Improves on the bipartite seed by performing a local search to add 
    additional edges (chords) while maintaining the constraint that no 
    3-edge path has endpoints of equal degree.
    """
    n = 15
    # Map edge (u, v) with u < v to an index 0..104
    # We can just use a set of frozensets or tuples for simplicity first, 
    # but bitmask is faster.
    
    # Let's use a list of edges and a helper to check validity.
    # Initial state: K_{7,8}
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
            
    # Helper to get neighbors
    def get_neighbors(edges_set):
        neigh = [set() for _ in range(n)]
        for u, v in edges_set:
            neigh[u].add(v)
            neigh[v].add(u)
        return neigh

    def get_degrees(neigh):
        return [len(neigh[i]) for i in range(n)]

    def is_valid(edges_set):
        neigh = get_neighbors(edges_set)
        degs = get_degrees(neigh)
        # Check all paths s-u-v-t
        # s in neigh[u], t in neigh[v]
        # s != t, s != v, t != u
        # Also u != v (guaranteed by simple graph)
        # Path is simple if s, u, v, t are distinct.
        # Since it's a simple graph, u!=v.
        # s != u (neighbor), t != v (neighbor).
        # We need s != t, s != v, t != u.
        for u, v in edges_set:
            for s in neigh[u]:
                if s == v: continue
                for t in neigh[v]:
                    if t == u or t == s: continue
                    if degs[s] == degs[t]:
                        return False
        return True

    # Local Search
    deadline = time.time() + 50.0 # 50s margin
    
    current_edges = edges
    current_valid = is_valid(current_edges)
    
    # Greedy addition
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in all_edges:
            if e in current_edges:
                continue
            # Try adding e
            new_edges = set(current_edges)
            new_edges.add(e)
            if is_valid(new_edges):
                current_edges = new_edges
                improved = True
                # Restart loop or continue? 
                # Continuing is fine, but restarting might find more.
                # Let's break to restart the loop to pick up new opportunities
                break
        
        # If no single addition worked, try swaps
        if not improved and time.time() < deadline:
            # Try removing one edge and adding another
            # This is O(E^2) which is ~11000 checks. 
            # Each check is O(E*N^2) ~ 23k ops. Total ~ 250M ops.
            # This might be too slow in Python if done naively.
            # We can optimize the validity check or limit the swaps.
            
            # Optimization: Only try swaps where the added edge is "promising".
            # Or just try a random subset of swaps.
            
            # Let's try a limited number of random swaps to avoid timeout
            import random
            attempts = 0
            max_attempts = 5000 # Adjust based on speed
            
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                # Pick a random edge to remove
                if not current_edges: break
                rem_e = random.choice(list(current_edges))
                # Pick a random edge to add (not in current)
                missing = [e for e in all_edges if e not in current_edges]
                if not missing: break
                add_e = random.choice(missing)
                
                new_edges = set(current_edges)
                new_edges.remove(rem_e)
                new_edges.add(add_e)
                
                if is_valid(new_edges):
                    current_edges = new_edges
                    improved = True
                    break # Restart swap search? Or continue?
                    # Breaking allows us to restart the greedy phase too.
                    # But we are in the "if not improved" block of the outer while.
                    # So we will loop back to greedy addition.
                    
    # Convert to list of lists
    result = [[u, v] for u, v in current_edges]
    return result
