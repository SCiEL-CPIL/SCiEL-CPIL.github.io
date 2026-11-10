"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 16}

def _diameter_and_critical(n, adj):
    """Compute diameter and check if every edge is critical.
    adj: list of sets.
    Returns (diameter, all_critical) where all_critical is True if every edge
    is critical (removal increases diameter above 2 or disconnects).
    """
    # BFS from each vertex to get all-pairs distances
    dist = [[-1]*n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)
    # Check connectivity and diameter
    diam = 0
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] == -1:
                return -1, False  # disconnected
            if dist[i][j] > diam:
                diam = dist[i][j]
    if diam > 2:
        return diam, False
    
    # Check if every edge is critical
    # An edge (u,v) is critical if removing it makes diameter > 2 or disconnects
    for u in range(n):
        for v in list(adj[u]):
            if v <= u:
                continue
            # Temporarily remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            
            # BFS to check diameter
            disconnected = False
            new_diam = 0
            for s in range(n):
                d = [-1]*n
                d[s] = 0
                q = deque([s])
                while q:
                    uu = q.popleft()
                    for vv in adj[uu]:
                        if d[vv] == -1:
                            d[vv] = d[uu] + 1
                            q.append(vv)
                for t in range(n):
                    if d[t] == -1:
                        disconnected = True
                        break
                    if d[t] > new_diam:
                        new_diam = d[t]
                if disconnected:
                    break
            
            # Restore edge
            adj[u].add(v)
            adj[v].add(u)
            
            if disconnected or new_diam > 2:
                # Edge is critical - good
                pass
            else:
                # Edge is NOT critical - bad
                return diam, False
    return diam, True

def _is_valid(n, adj):
    """Check if graph has diameter exactly 2 and every edge is critical."""
    # First check diameter is exactly 2
    diam, _ = _diameter_and_critical(n, adj)
    if diam != 2:
        return False
    # Now check all edges critical
    _, all_crit = _diameter_and_critical(n, adj)
    return all_crit

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = P['n']
    
    # Strategy: Start with a random graph, try to find a valid one
    # Better: Start from K16 minus some edges, iteratively optimize
    
    best_edges = 0
    best_adj = None
    
    # Try multiple random initializations
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Start with a random graph that has diameter 2
        # Ensure connectivity and diameter 2
        adj = [set() for _ in range(n)]
        
        # Create a spanning path/tree for connectivity
        random.seed(trial * 1000 + 42)
        order = list(range(n))
        random.shuffle(order)
        for i in range(n-1):
            u, v = order[i], order[i+1]
            adj[u].add(v)
            adj[v].add(u)
        
        # Add random edges to get diameter 2
        all_pairs = [(i,j) for i in range(n) for j in range(i+1,n) if j not in adj[i]]
        random.shuffle(all_pairs)
        for u,v in all_pairs:
            if time.time() > deadline - 10:
                break
            adj[u].add(v)
            adj[v].add(u)
        
        # Now we have a dense graph. Remove edges greedily while maintaining validity
        # But first, check if diameter is exactly 2
        # Remove edges that are non-critical
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            # Collect edges
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u,v))
            random.shuffle(edges)
            for u,v in edges:
                if time.time() > deadline - 5:
                    break
                if v not in adj[u]:
                    continue
                # Try removing
                adj[u].discard(v)
                adj[v].discard(u)
                # Check validity
                valid = _is_valid(n, adj)
                if not valid:
                    # Restore
                    adj[u].add(v)
                    adj[v].add(u)
                else:
                    changed = True
        
        # After removal, check if valid
        if _is_valid(n, adj):
            edge_count = sum(len(adj[u]) for u in range(n)) // 2
            if edge_count > best_edges:
                best_edges = edge_count
                best_adj = [set(s) for s in adj]
        
        # Now try to ADD edges to the best graph found so far
        if best_adj is not None:
            adj = [set(s) for s in best_adj]
            improved = True
            while improved:
                if time.time() > deadline - 5:
                    break
                improved = False
                # Try adding each missing edge
                missing = [(i,j) for i in range(n) for j in range(i+1,n) if j not in adj[i]]
                random.shuffle(missing)
                for u,v in missing:
                    if time.time() > deadline - 5:
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                    if _is_valid(n, adj):
                        improved = True
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)
            
            if _is_valid(n, adj):
                edge_count = sum(len(adj[u]) for u in range(n)) // 2
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = [set(s) for s in adj]
    
    # If no valid graph found, return a fallback
    if best_adj is None:
        # Fallback: a known valid graph
        # Cycle C16 has diameter 8, not good.
        # Use a different construction
        # Actually, let's just return a star (diameter 2, but not all edges critical)
        # Better fallback: return empty or a simple valid graph
        # For n=16, a valid graph: take K16 and remove a perfect matching
        # But need to verify...
        # Just return a minimal valid graph as fallback
        adj = [set() for _ in range(n)]
        # Complete bipartite K(8,8) has diameter 2, but edges are not all critical
        # Let's just return the best we have or a default
        return [[0,v] for v in range(1,n)]
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
