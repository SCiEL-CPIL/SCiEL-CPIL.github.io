"""Enhanced: Grötzsch core + local search for triangle-free 4-chromatic graph."""

import time
import random

_DEADLINE = None

def _time_ok():
    global _DEADLINE
    if _DEADLINE is None:
        return True
    return time.time() < _DEADLINE

def _is_triangle_free(adj, n):
    """Check if graph is triangle-free."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    return True

def _is_3_colorable(adj, n):
    """
    Fast 3-colorability check using bitsets and backtracking.
    Returns True if 3-colorable.
    """
    # Pre-compute neighbor masks for fast forbidden color check
    # adj_mask[v] is a bitmask of neighbors of v
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)

    # Order vertices by degree descending
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    # Map vertex to its index in the order
    pos = {v: i for i, v in enumerate(order)}

    # color[v] = 0, 1, 2 or -1 if uncolored
    color = [-1] * n

    # Backtracking function
    def backtrack(idx):
        if idx == n:
            return True

        v = order[idx]
        
        # Determine forbidden colors
        # A color c is forbidden if any neighbor of v already has color c
        # We can check this by looking at colored neighbors.
        # To optimize, we can maintain a bitmask of colored neighbors for each color?
        # Or just iterate. n=16 is small.
        
        forbidden = 0
        for nb in adj[v]:
            if pos[nb] < idx:
                forbidden |= (1 << color[nb])
        
        # Try each available color
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False

    return backtrack(0)

def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) preserves triangle-free property."""
    if v in adj[u]:
        return False
    # Check if any common neighbor exists
    # If w is neighbor of u and w is neighbor of v, then triangle u-v-w
    for w in adj[u]:
        if w in adj[v]:
            return False
    return True

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115  # Safety margin
    
    random.seed(42)
    n = 16
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    # 1. Initialize with Grötzsch Graph (11 vertices)
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    
    # Mycielski layer 5-9
    for i in range(5):
        left = (i - 1) % 5
        right = (i + 1) % 5
        add_edge(5 + i, left)
        add_edge(5 + i, right)
    
    # Top vertex 10
    for i in range(5):
        add_edge(10, 5 + i)

    # 2. Add vertices 11-15
    # Strategy: Connect new vertex to a large independent set of existing graph
    # to maximize edges while keeping it triangle-free (independent set in G means clique in complement, 
    # but here we just need non-adjacent vertices in G so no triangles formed with new vertex).
    
    for v in range(11, 16):
        if not _time_ok():
            break
        
        # Find a large independent set in the subgraph induced by 0..10
        # Greedy MIS approximation
        existing = list(range(11))
        random.shuffle(existing)
        
        indep = []
        for u in existing:
            if all(u not in adj[w] for w in indep):
                indep.append(u)
        
        # Add edges from v to indep
        for u in indep:
            add_edge(v, u)
            
        # Also try to connect v to other vertices if possible?
        # No, if u and w are both in indep, they are not adjacent. 
        # If we add v-u and v-w, no triangle v-u-w because u-w not edge.
        # Can we add more edges?
        # If we add v-z where z is NOT in indep, we must ensure z is not adjacent to any u in indep.
        # Actually, if z is adjacent to u in indep, then v-z and v-u and u-z form triangle.
        # So v can only connect to vertices that form an independent set in the "conflict" graph?
        # Simpler: The set of neighbors of v must be an independent set in G.
        # So we just need to pick a maximal independent set.
        # The greedy above finds one. We can try to improve it.
        
        # Try to improve the independent set for v
        # Local search on the set of neighbors of v
        # This is effectively finding a Max Independent Set in the subgraph of existing vertices.
        # Since n is small (11), we can do a better job.
        
        # Let's try a simple local improvement on the neighbor set of v
        changed = True
        while changed and _time_ok():
            changed = False
            # Try to add a vertex
            for u in range(11):
                if u in adj[v]:
                    continue
                if all(u not in adj[w] for w in adj[v]):
                    add_edge(v, u)
                    changed = True
                    break
            # Try to swap? (Remove one, add two?) - complex, skip for now.
            # The "add" step is sufficient for now as we start from a decent greedy set.

    # 3. Local Search Phase: Try to add any missing edge
    # Iterate until no more edges can be added or time runs out
    improved = True
    while improved and _time_ok():
        improved = False
        # Shuffle edges to avoid bias
        # Generate all potential edges
        potential_edges = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            if not _time_ok():
                break
            
            if v in adj[u]:
                continue
                
            # Check triangle-free
            if not _can_add_edge(adj, u, v):
                continue
            
            # Check 3-colorability
            # Temporarily add edge
            adj[u].add(v)
            adj[v].add(u)
            
            if not _is_3_colorable(adj, n):
                # Keep edge
                improved = True
                break # Restart search with new graph state
            else:
                # Remove edge
                adj[u].remove(v)
                adj[v].remove(u)
        
        # If we broke out due to time, we should stop the outer loop too
        if not _time_ok():
            break

    # Collect edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                edges.append([u, v])
    
    # Final verification
    if not _is_triangle_free(adj, n):
         # Fallback to seed
         return [[0,1],[1,2],[2,3],[3,4],[4,0],[0,6],[1,7],[2,8],[3,9],[4,10],[5,6],[5,7],[5,8],[5,9],[5,10]]
    
    if _is_3_colorable(adj, n):
         # Fallback to just Grötzsch + isolated
         edges = []
         adj2 = [set() for _ in range(n)]
         for i in range(5):
             adj2[i].add((i+1)%5)
             adj2[(i+1)%5].add(i)
         for i in range(5):
             left = (i-1)%5
             right = (i+1)%5
             adj2[5+i].add(left)
             adj2[left].add(5+i)
             adj2[5+i].add(right)
             adj2[right].add(5+i)
         for i in range(5):
             adj2[10].add(5+i)
             adj2[5+i].add(10)
         for u in range(n):
             for v in adj2[u]:
                 if v > u:
                     edges.append([u, v])
         return edges
    
    return edges
