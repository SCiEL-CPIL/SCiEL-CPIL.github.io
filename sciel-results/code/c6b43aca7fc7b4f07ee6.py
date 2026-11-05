import random
import time
import sys

def _has_triangle(adj, n):
    """Check if graph has a triangle. adj is list of sets."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                # Check intersection of adj[u] and adj[v]
                # If there is a w > v in both, it's a triangle
                # Simple check:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return (u, v, w)
    return None

def _find_independent_set(adj, n, k):
    """Find an independent set of size k using DFS with pruning."""
    # This is a simple backtracking search
    # For n=40, this might be slow if alpha is large, but we only need one set of size 10.
    # We can use a greedy heuristic to find a large independent set first.
    
    # Greedy independent set
    candidates = set(range(n))
    ind_set = []
    while candidates:
        # Pick vertex with min degree in induced subgraph? Or random?
        # For speed, pick random
        v = random.choice(list(candidates))
        ind_set.append(v)
        candidates -= adj[v]
        candidates.discard(v)
        
    if len(ind_set) >= k:
        return ind_set[:k]
        
    # If greedy fails, we might need exact search, but for n=40 and alpha < 10 is the goal,
    # if greedy finds < 10, it's likely alpha < 10.
    # However, greedy can fail to find a large set if one exists.
    # But if we are in a state where we *think* alpha >= 10, we need to be sure.
    # For the purpose of this task, if greedy finds < 10, we assume valid.
    # If greedy finds >= 10, we use it to fix.
    # This is a heuristic.
    
    return None

def _validate(adj, n):
    """Check if graph is triangle-free and alpha < 10."""
    # Check triangles
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    # Check independent set size 10
    # Use a quick branch and bound for max independent set
    # Or just check if we can find one of size 10.
    
    # Simple DFS for independent set of size 10
    def dfs(candidates, current_size):
        if current_size == 10:
            return True
        if not candidates:
            return False
        # Pruning: if current_size + len(candidates) < 10, return False
        if current_size + len(candidates) < 10:
            return False
            
        # Heuristic: pick vertex with lowest degree in candidates
        # For simplicity, pick first
        v = next(iter(candidates))
        
        # Option 1: Include v
        new_candidates = candidates - adj[v]
        new_candidates.discard(v)
        if dfs(new_candidates, current_size + 1):
            return True
            
        # Option 2: Exclude v
        new_candidates2 = candidates - {v}
        if dfs(new_candidates2, current_size):
            return True
            
        return False
        
    if dfs(set(range(n)), 0):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to construct for n=40
    n = 40
    best_graph = None
    
    while time.time() < deadline:
        # Initialize random graph
        adj = [set() for _ in range(n)]
        # Random edges with p=0.2
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.2:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Remove triangles
        # Loop until no triangles
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in list(adj[u]):
                    if v > u:
                        # Check for common neighbors w > v
                        common = adj[u].intersection(adj[v])
                        common = {w for w in common if w > v}
                        if common:
                            # Remove a random edge from the triangle
                            w = random.choice(list(common))
                            # Remove edge (u, w) or (v, w) or (u, v)
                            # Remove (u, v) to break triangle
                            adj[u].discard(v)
                            adj[v].discard(u)
                            changed = True
                            break
                if changed:
                    break
            if changed:
                continue
            # Check for triangles again in next outer loop? 
            # The inner loop breaks on first removal.
        
        # Check validity
        if _validate(adj, n):
            best_graph = adj
            break
            
        # If invalid, try local search to fix independent set
        # Find independent set of 10
        ind_set = _find_independent_set(adj, n, 10)
        if ind_set:
            # Try to add edges in ind_set
            improved = False
            for u in ind_set:
                for v in ind_set:
                    if u < v and v not in adj[u]:
                        # Check if adding (u, v) creates a triangle
                        # Common neighbors of u and v
                        common = adj[u].intersection(adj[v])
                        if not common:
                            # Safe to add
                            adj[u].add(v)
                            adj[v].add(u)
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                # Re-check validity after improvement
                if _validate(adj, n):
                    best_graph = adj
                    break
        else:
            # Greedy didn't find size 10, but _validate might have found one?
            # _validate is exact. If _validate returned False, it found one.
            # So if _validate returned True, we would have broken.
            # So here, _validate returned False, meaning there IS an independent set of 10.
            # But _find_independent_set (greedy) failed?
            # This can happen. In that case, we restart.
            pass

    if best_graph:
        edges = []
        for i in range(n):
            for j in best_graph[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # Fallback to n=39
    n = 39
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.2:
                adj[i].add(j)
                adj[j].add(i)
                
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in list(adj[u]):
                if v > u:
                    common = adj[u].intersection(adj[v])
                    common = {w for w in common if w > v}
                    if common:
                        w = random.choice(list(common))
                        adj[u].discard(v)
                        adj[v].discard(u)
                        changed = True
                        break
            if changed:
                break
                
    if _validate(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # Fallback to Andrasfai n=26
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

# EVOLVE-BLOCK-END
