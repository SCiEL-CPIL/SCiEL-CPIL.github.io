import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_diameter(adj, n):
    """
    Returns the diameter of the graph.
    """
    max_diam = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        
        if any(d == -1 for d in dist):
            return 999
            
        for d in dist:
            if d > max_diam:
                max_diam = d
        
    return max_diam

def _is_valid_solution(added_edges, n, base_edges):
    """
    Checks if the added edges result in a valid graph:
    1. Triangle-free
    2. Connected
    3. Diameter <= 4
    """
    adj = {i: set() for i in range(n)}
    
    # Add base edges
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Add new edges and check for duplicates
    for u, v in added_edges:
        if u in adj[v]:
            return False # Duplicate or self-loop (though self-loop is impossible with u!=v)
        adj[u].add(v)
        adj[v].add(u)
        
    # Check Triangle-Free
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    return False
                    
    # Check Connected and Diameter
    diam = _bfs_diameter(adj, n)
    if diam > 4:
        return False
        
    return True

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # 1. Deterministic Candidate Construction (from Inspiration 1)
    # We exploit the bipartite nature of the path graph.
    # Any edge connecting an even vertex to an odd vertex is triangle-free with the base path.
    # We will search for a minimal subset of such edges that reduces diameter to <= 4.
    
    # Pre-compute all valid candidate edges (cross-parity, not in base)
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2:
                all_candidates.append((u, v))
                
    # Sort candidates by "potential" - edges that connect vertices far apart in the path
    # are likely more useful for reducing diameter.
    # Distance in path is abs(u - v).
    all_candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    best_solution = None
    best_count = float('inf')
    
    deadline = time.time() + 85
    
    # 2. Heuristic Search (Refinement from Current Program)
    # We use a randomized greedy approach with backtracking/branching on the top candidates.
    # Since the search space is small (24 nodes), we can be relatively aggressive.
    
    # First, try a deterministic greedy pass
    # We pick the candidate that most reduces diameter, breaking ties by path distance.
    
    def greedy_solve(candidate_order, max_depth=10):
        """
        Greedy search.
        """
        current_adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        curr_diam = _bfs_diameter(current_adj, n)
        if curr_diam <= 4:
            return []
            
        current_added = []
        
        # We will try to add edges one by one.
        # To keep it fast, we limit the branching factor and depth.
        # Since we need to find a *minimal* set, we can use a simple greedy with a twist:
        # At each step, we evaluate the top K candidates and pick the best one.
        
        remaining_candidates = list(candidate_order)
        
        while curr_diam > 4 and remaining_candidates:
            best_edge = None
            best_new_diam = curr_diam
            
            # Evaluate top 20 candidates to save time
            for u, v in remaining_candidates[:20]:
                if v in current_adj[u]:
                    continue
                    
                # Tentatively add
                current_adj[u].add(v)
                current_adj[v].add(u)
                
                new_diam = _bfs_diameter(current_adj, n)
                
                # Undo
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                    
            if best_edge is None:
                # No improvement found in top candidates?
                # Try the rest? Or break?
                # For now, if top 20 don't help, maybe we need a different order.
                # But if we are stuck, we return None to signal failure for this path.
                break
                
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_added.append((u, v))
            curr_diam = best_new_diam
            
            # Remove used edge from candidates
            # Note: remaining_candidates is a list of tuples.
            # We just remove it.
            remaining_candidates = [e for e in remaining_candidates if e != (u, v)]
            
        if curr_diam <= 4:
            return current_added
        else:
            return None

    # Run greedy with the sorted candidates
    sol = greedy_solve(all_candidates)
    if sol and _is_valid_solution(sol, n, base_edges):
        best_solution = sol
        best_count = len(sol)
        
    # If the deterministic greedy didn't find a small solution, or to find a better one,
    # run randomized restarts.
    # We shuffle the candidates and run greedy again.
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        # Shuffle candidates, but keep the high-potential ones somewhat prioritized?
        # Or just shuffle fully to explore different local optima.
        shuffled_candidates = all_candidates[:]
        random.shuffle(shuffled_candidates)
        
        sol = greedy_solve(shuffled_candidates)
        if sol and _is_valid_solution(sol, n, base_edges):
            if len(sol) < best_count:
                best_count = len(sol)
                best_solution = sol
                
                # If we found a very small solution, we can stop early?
                # Theoretical minimum is likely around 3-5 edges.
                if best_count <= 3:
                    break
                    
    if best_solution is None:
        # Fallback: Return a known valid set if we failed (shouldn't happen with N=24)
        # The hub strategy from Inspiration 1 is a safe fallback.
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        fallback = [[hub, v] for v in targets]
        if _is_valid_solution(fallback, n, base_edges):
            return [[int(u), int(v)] for u, v in fallback]
        else:
            return []
            
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
