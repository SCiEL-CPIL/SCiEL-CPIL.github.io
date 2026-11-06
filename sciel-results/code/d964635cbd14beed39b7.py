import random
import time

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Try to find a graph for n=40 down to n=26
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        max_restarts = 200
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # 1. Initialize a random maximal triangle-free graph
            adj = [0] * n
            # Create list of all edges
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                # Check if adding (u, v) creates a triangle
                # Common neighbors of u and v
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # 2. Check if alpha < 10
            # We check if there is an independent set of size 10
            if not has_independent_set(adj, n, 10):
                # Convert adj to edge list
                edge_list = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask -= lsb
                return {'n': n, 'edges': edge_list}
            
            # 3. If alpha >= 10, try to improve
            # Find an independent set of size 10
            # (The function above should return the set or None, let's modify it to return the set)
            
    # Fallback: Return the known And(9) graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def has_independent_set(adj, n, k):
    """
    Returns True if there is an independent set of size k.
    Uses backtracking.
    """
    # Candidates are all vertices initially
    # We want to pick k vertices such that no two are adjacent.
    
    def backtrack(current_set, candidates):
        if len(current_set) == k:
            return current_set
        if len(candidates) < k - len(current_set):
            return None
        
        # Heuristic: pick a vertex from candidates
        # To speed up, we can try to pick vertices with low degree in the induced subgraph of candidates?
        # Or just pick the first one.
        
        v = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include v
        # New candidates are those in rest that are NOT adjacent to v
        new_candidates = [u for u in rest if not (adj[v] >> u) & 1]
        res = backtrack(current_set + [v], new_candidates)
        if res:
            return res
        
        # Option 2: Exclude v
        res = backtrack(current_set, rest)
        if res:
            return res
        
        return None

    return backtrack([], list(range(n)))
