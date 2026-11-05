import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We want to maximize n.
    # Try n from 47 down to 1.
    best_result = {'n': 39, 'edges': []} # Fallback to seed
    
    for n in range(47, 0, -1):
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # If min_deg is 0, empty graph works.
        if min_deg == 0:
            return {'n': n, 'edges': []}
            
        # Try to find a graph with min degree >= min_deg and no C4.
        # We aim for regular degree min_deg if possible, or just min_deg.
        # Greedy approach:
        # We run multiple attempts.
        attempts = 0
        max_attempts = 2000  # Adjust based on time
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            success, edges = _try_construct(n, min_deg)
            if success:
                return {'n': n, 'edges': edges}
                
        # If we didn't find it for this n, try smaller n
        # But we must ensure we return the best found so far if we run out of time?
        # The loop decreases n, so the first success is the best.
        # If we fail all n, we return the seed.
        
    return best_result

def _try_construct(n, min_deg):
    # Initialize empty graph
    adj = [set() for _ in range(n)]
    degrees = [0] * n
    
    # List of vertices to process
    # We want to raise degrees to at least min_deg
    
    # Randomize vertex order for processing
    vertices = list(range(n))
    random.shuffle(vertices)
    
    # We will iteratively add edges.
    # To avoid getting stuck, we can use a priority queue or just pick random unsatisfied vertices.
    
    for _ in range(n * min_deg * 2): # Max steps heuristic
        # Find vertices with degree < min_deg
        unsatisfied = [v for v in range(n) if degrees[v] < min_deg]
        if not unsatisfied:
            return True, [[u, v] for u in range(n) for v in adj[u] if u < v]
        
        # Pick a random unsatisfied vertex
        u = random.choice(unsatisfied)
        
        # Find candidates for v
        candidates = []
        for v in range(n):
            if v == u or v in adj[u]:
                continue
            # Check C4 constraint: |N(u) \cap N(v)| <= 1
            # If we add uv, we need to ensure no C4 is formed.
            # A C4 is formed if there are x,y such that u-x-v-y-u?
            # No, adding uv creates C4 if there is a path u-x-v of length 2?
            # If u-x-v exists, adding uv creates triangle u-x-v-u. OK.
            # If u-x-y-v exists, adding uv creates C4 u-x-y-v-u.
            # So we need to check if there are two distinct common neighbors?
            # Wait.
            # If N(u) \cap N(v) has size 2, say {x, y}.
            # Then u-x-v and u-y-v are paths of length 2.
            # Adding uv creates two triangles? No.
            # The cycle is u-x-v-y-u? No, x and y are neighbors of both.
            # The cycle is u-x-v-y-u requires x-y edge.
            # The C4 is u-x-v-z-u where z is another common neighbor?
            # If x, y are common neighbors, then u-x-v and u-y-v.
            # The cycle u-x-v-y-u is not necessarily a cycle unless x-y is an edge.
            # The cycle is u-x-v-w-u?
            # Actually, if x, y are common neighbors, the subgraph induced by {u,v,x,y} has edges ux, vx, uy, vy.
            # This is a K_{2,2} which is a C4.
            # So yes, if |N(u) \cap N(v)| >= 2, adding uv creates a C4.
            
            common = adj[u].intersection(adj[v])
            if len(common) <= 1:
                # Also check if adding uv creates C4 with other nodes?
                # No, only common neighbors matter for immediate C4 formation involving uv.
                candidates.append(v)
                
        if not candidates:
            # Stuck. Could try swapping, but for now just fail this attempt.
            return False, []
            
        # Pick a candidate, prefer one with low degree to balance
        # Sort by degree
        candidates.sort(key=lambda x: degrees[x])
        v = candidates[0]
        
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        degrees[u] += 1
        degrees[v] += 1

    # Check if all satisfied
    if all(degrees[v] >= min_deg for v in range(n)):
        return True, [[u, v] for u in range(n) for v in adj[u] if u < v]
        
    return False, []
