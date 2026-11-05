import time
import random

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # We know n=46 is possible. Try 47 first, then 46, etc.
    # The task says value is known to be 46 or 47.
    # If we find 47, score is higher.
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Try to construct for this n
        result = _fast_greedy_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}

    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _is_safe(adj, u, v):
    """
    Check if adding edge (u,v) to graph defined by adjacency sets adj creates a C4.
    adj is a list of sets.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We need to check if there is an edge (x,y) in E
    # such that x in N(u), y in N(v), and x != y.
    # This creates cycle u-x-y-v-u.
    
    for x in nu:
        # Check if any neighbor of x is in nv
        # If x is in nv, we must exclude x from the check for y
        # because y != x
        common = adj[x] & nv
        if common:
            # If x is in common, it means x is a neighbor of itself? No, x in nv and x in adj[x] is false.
            # Wait, adj[x] contains neighbors of x. x is not in adj[x].
            # So if x in nv, x is not in adj[x].
            # So common will not contain x.
            # Therefore, if common is not empty, there is some y in nv such that (x,y) in E.
            # And since x not in adj[x], y != x is automatically satisfied?
            # No, y is in nv. x is in nu.
            # If x is in nv, then x is a common neighbor.
            # But x is not in adj[x]. So x is not in common.
            # So any y in common is distinct from x.
            return False
            
    return True

def _fast_greedy_construct(n, required_min_deg, deadline):
    """
    Attempts to build a C4-free graph with min degree >= required_min_deg.
    Uses multiple random restarts with a greedy strategy.
    """
    max_restarts = 5000  # Set-based check is fast, so we can do many restarts
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
            
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        
        success = False
        
        # We need to add edges until min(degrees) >= required_min_deg
        # Total edges needed approx n * required_min_deg / 2
        
        while time.time() < deadline:
            min_deg = min(degrees)
            if min_deg >= required_min_deg:
                success = True
                break
                
            # Find vertices with min degree
            min_vertices = [i for i in range(n) if degrees[i] == min_deg]
            
            # Shuffle to try random order
            random.shuffle(min_vertices)
            
            added_any = False
            for u in min_vertices:
                # Get non-neighbors of u
                non_neighbors = [v for v in range(n) if v not in adj[u] and v != u]
                
                if not non_neighbors:
                    continue
                    
                # Shuffle to try random order
                random.shuffle(non_neighbors)
                
                for v in non_neighbors:
                    if _is_safe(adj, u, v):
                        # Add edge
                        adj[u].add(v)
                        adj[v].add(u)
                        degrees[u] += 1
                        degrees[v] += 1
                        added_any = True
                        break  # Restart loop to re-evaluate degrees
                
                if added_any:
                    break  # Restart outer while loop
            
            if not added_any:
                # Could not add any edge to any min-degree vertex
                # Try a repair step: remove a random edge and try again
                if not _repair_step(adj, degrees, n, deadline):
                    break
                # After repair, continue the while loop
                
        if success:
            edges = []
            for u in range(n):
                for v in sorted(adj[u]):
                    if u < v:
                        edges.append([u, v])
            return (n, edges)
            
    return None

def _repair_step(adj, degrees, n, deadline):
    """
    Try to unblock by removing a random edge and letting the greedy continue.
    Returns True if an edge was removed (making progress possible), False otherwise.
    """
    # Collect all edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if not edges:
        return False
    
    # Try removing up to 3 random edges
    for _ in range(3):
        if time.time() > deadline:
            return False
        
        idx = random.randrange(len(edges))
        u, v = edges[idx]
        
        # Remove edge
        adj[u].discard(v)
        adj[v].discard(u)
        degrees[u] -= 1
        degrees[v] -= 1
        
        # Now try to add a different edge to one of the min-degree vertices
        min_deg = min(degrees)
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        random.shuffle(min_vertices)
        
        for u2 in min_vertices:
            non_neighbors = [v2 for v2 in range(n) if v2 not in adj[u2] and v2 != u2]
            random.shuffle(non_neighbors)
            
            for v2 in non_neighbors:
                if _is_safe(adj, u2, v2):
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    degrees[u2] += 1
                    degrees[v2] += 1
                    return True
    
    return False
