import itertools
import time

def _check_triangle_free(n, D):
    # D is a set of differences
    D_set = set(D)
    for a in D_set:
        for b in D_set:
            s = a + b
            if s in D_set:
                return False
            if (n - s) in D_set:
                return False
    return True

def _has_independent_set_size_10(n, D):
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for d in D:
            j = (i + d) % n
            adj[i].add(j)
            k = (i - d) % n
            adj[i].add(k)
            
    # Branch and bound to find independent set of size 10
    # candidates is a list of vertices
    # current is the set of vertices in the independent set so far
    
    def dfs(candidates, current_size):
        if current_size == 10:
            return True
        if not candidates:
            return False
        if current_size + len(candidates) < 10:
            return False
            
        # Heuristic: pick vertex with max degree in candidate subgraph?
        # Or just first one.
        # To speed up, we can sort candidates by degree in the induced subgraph
        # But for n=40, simple selection might suffice.
        
        # Pick a vertex
        # We want to find a set of size 10.
        # If we pick a vertex v, we remove its neighbors.
        # If we don't pick v, we remove v.
        
        # Order: Try including high-degree vertices first? 
        # Actually, to find an independent set, we should pick vertices that are "far apart".
        # A simple heuristic: pick the vertex with the smallest degree in the remaining graph?
        # No, if we pick a vertex with many neighbors, we eliminate many candidates, 
        # which helps prune the "include" branch.
        
        # Let's just pick the first one for simplicity, or randomize?
        # Deterministic is better for reproducibility.
        v = candidates[0]
        rest = candidates[1:]
        
        # Branch 1: Include v
        # New candidates: rest - neighbors(v)
        new_candidates = [u for u in rest if u not in adj[v]]
        if dfs(new_candidates, current_size + 1):
            return True
            
        # Branch 2: Exclude v
        if dfs(rest, current_size):
            return True
            
        return False

    return dfs(list(range(n)), 0)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Try n from 40 down to 35
    for n in range(40, 34, -1):
        if time.time() > deadline:
            break
            
        # Try difference sets of size 4 first (max degree 8)
        # Then size 3, etc.
        # We need triangle-free.
        # We need alpha < 10.
        
        # Generate combinations of differences
        # d in 1..n//2
        max_d = n // 2
        diffs = range(1, max_d + 1)
        
        found = False
        # Try k=4
        for D in itertools.combinations(diffs, 4):
            if time.time() > deadline:
                break
            if not _check_triangle_free(n, D):
                continue
            if _has_independent_set_size_10(n, D):
                continue
            
            # Valid graph found
            edges = []
            for i in range(n):
                for d in D:
                    j = (i + d) % n
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        # Try k=3
        for D in itertools.combinations(diffs, 3):
            if time.time() > deadline:
                break
            if not _check_triangle_free(n, D):
                continue
            if _has_independent_set_size_10(n, D):
                continue
            
            edges = []
            for i in range(n):
                for d in D:
                    j = (i + d) % n
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        # Try k=2
        for D in itertools.combinations(diffs, 2):
            if time.time() > deadline:
                break
            if not _check_triangle_free(n, D):
                continue
            if _has_independent_set_size_10(n, D):
                continue
            
            edges = []
            for i in range(n):
                for d in D:
                    j = (i + d) % n
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback to seed if nothing found (should not happen)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
