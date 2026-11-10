"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    # Construct 2 disjoint K_{3,3} components
    # Component 1: Vertices 0..5
    # Component 2: Vertices 6..11
    
    edges = set()
    
    # K_{3,3} on 0..5: U={0,1,2}, V={3,4,5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.add(tuple(sorted((u, v))))
            
    # K_{3,3} on 6..11: U={6,7,8}, V={9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.add(tuple(sorted((u, v))))
            
    # Convert to list of lists
    result = [list(e) for e in sorted(edges)]
    
    # Optional: Verify validity if time permits (fast for this size)
    if time.time() < deadline:
        # Check min degree
        deg = [0] * 12
        for u, v in result:
            deg[u] += 1
            deg[v] += 1
        if min(deg) < 3:
            # Fallback to seed (though it should be valid)
            pass
            
        # Check triangle-free
        adj = {i: set() for i in range(12)}
        for u, v in result:
            adj[u].add(v)
            adj[v].add(u)
            
        for u in range(12):
            for v in adj[u]:
                if v < u: continue
                for w in adj[v]:
                    if w in adj[u] and w != u and w != v:
                        # Triangle found
                        pass
                        
    return result

# EVOLVE-BLOCK-END
