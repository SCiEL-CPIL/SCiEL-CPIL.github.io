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
            return False # Duplicate or self-loop
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
    
    # 1. Precompute base graph properties
    base_adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    # Compute distances in base graph (which is a path 0-23)
    # dist_base[u][v] = abs(u - v)
    # Pairs that are > 4 apart in the base path are the ones we need to "cover"
    # such that in the augmented graph, their distance becomes <= 4.
    
    # Actually, it's not just pairs > 4 apart. It's pairs that might end up > 4 apart
    # even if they are close in the path, if we don't add shortcuts?
    # No, the path distance is the upper bound. If we add edges, distances only decrease.
    # So if dist_base(u,v) <= 4, they are already fine.
    # We only need to worry about pairs where dist_base(u,v) > 4.
    
    # Let's identify the "critical pairs" that must have distance <= 4 in the final graph.
    # These are all pairs (u,v) such that abs(u-v) > 4.
    
    # Strategy: Set Cover / Branch and Bound
    # Universe U: Pairs (u,v) with abs(u-v) > 4.
    # Sets S_e for each candidate edge e=(a,b):
    #   S_e = { (u,v) in U | adding edge e reduces dist(u,v) to <= 4 }
    # Wait, "reduces dist to <= 4" depends on the *combination* of edges.
    # This is not a standard set cover because the effect of an edge depends on others.
    # However, we can use a heuristic: An edge (a,b) helps pair (u,v) if it provides a shortcut.
    
    # A more robust approach for small N=24:
    # Generate all valid bipartite edges (even-odd).
    # Use a Branch and Bound search to find the minimum set of edges.
    # State: (current_edges, current_diameter, index_of_next_candidate)
    # This is exponential, but with good pruning (if current_diameter <= 4, we have a solution),
    # and N=24 is small enough that the number of "useful" edges might be small.
    
    # Let's refine the "useful" edges.
    # Only edges that connect vertices with distance > 4 in the path are potentially useful
    # for reducing the diameter below the path distance.
    # Edges with dist <= 4 don't help reduce the global diameter if the global diameter is > 4.
    # Actually, adding an edge (u,v) with dist_path(u,v) <= 4 doesn't change the distance between any pair (x,y) to be smaller than dist_path(x,y) in a way that helps the global diameter if the global diameter is determined by far pairs.
    # Wait, if I add (0,3), does it help (0,10)? No.
    # So we only consider edges (u,v) where abs(u-v) > 4.
    
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2: # Bipartite check
                if abs(u - v) > 4: # Only useful edges
                    candidates.append((u, v))
                    
    # Sort candidates by "potential".
    # A good heuristic is to sort by the number of pairs they "cover" in a greedy sense,
    # or just by path distance (longer edges are better).
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    best_solution = None
    best_count = float('inf')
    
    deadline = time.time() + 80 # Safety margin for 90s limit
    
    # Branch and Bound
    # We will try to build a solution edge by edge.
    # To speed up, we use a greedy lower bound:
    # The current diameter is D.
    # If we have k edges so far, and we need to reduce D to 4.
    # A simple bound: if k + 1 >= best_count, prune.
    
    # We also need a fast diameter check.
    # Instead of full BFS for every node, we can just check if diameter > 4.
    # For N=24, BFS is fast.
    
    def solve_bnb(candidates, base_adj, n, limit=4):
        # candidates is a list of (u,v)
        # We will use a recursive approach or iterative deepening.
        # Given the small size, let's try a simple DFS with pruning.
        
        # Pre-compute a "greedy" solution to set an upper bound
        # This helps prune the BnB.
        
        # Greedy Set Cover Approximation for initialization
        def greedy_init(cands):
            adj = {i: set(base_adj[i]) for i in range(n)}
            curr_diam = _bfs_diameter(adj, n)
            if curr_diam <= limit:
                return []
            
            added = []
            remaining_cands = cands[:]
            
            while curr_diam > limit and remaining_cands:
                best_edge = None
                best_new_diam = curr_diam
                
                # To save time, only check top 10 candidates?
                # Or all? N=24, candidates ~ 100-200.
                # Let's check all to be safe for the greedy step, it's fast.
                
                for e in remaining_cands:
                    u, v = e
                    if v in adj[u]:
                        continue
                    adj[u].add(v)
                    adj[v].add(u)
                    nd = _bfs_diameter(adj, n)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    if nd < best_new_diam:
                        best_new_diam = nd
                        best_edge = e
                        
                if best_edge is None:
                    return None # Stuck? Shouldn't happen if we have enough candidates
                    
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                added.append(e)
                curr_diam = best_new_diam
                remaining_cands = [e for e in remaining_cands if e != best_edge]
                
            if curr_diam <= limit:
                return added
            return None

        init_sol = greedy_init(candidates)
        if init_sol:
            best_count = len(init_sol)
            best_solution = init_sol
        else:
            # If greedy fails, we have a problem. But for N=24 path, it should work.
            # Fallback to a large hub solution
            hub = 11
            targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
            fallback = [[hub, v] for v in targets]
            if _is_valid_solution(fallback, n, base_edges):
                best_count = len(fallback)
                best_solution = fallback
            else:
                return [] # Should not happen

        # Now BnB to find smaller than best_count
        # We search for solutions of size 1, 2, 3... up to best_count - 1
        
        # To make BnB effective, we order candidates.
        # We already sorted by path distance.
        
        def dfs(start_idx, current_edges, current_adj, current_diam):
            nonlocal best_solution, best_count
            
            if time.time() > deadline:
                return
                
            if current_diam <= limit:
                if len(current_edges) < best_count:
                    best_count = len(current_edges)
                    best_solution = current_edges[:]
                return
                
            # Pruning: if we already have >= best_count edges, no point continuing
            if len(current_edges) >= best_count:
                return
                
            # Lower bound heuristic:
            # How many more edges do we *need*?
            # Hard to compute exactly.
            # Simple bound: 0.
            
            # Iterate through candidates from start_idx
            for i in range(start_idx, len(candidates)):
                e = candidates[i]
                u, v = e
                if v in current_adj[u]:
                    continue
                    
                # Add edge
                current_adj[u].add(v)
                current_adj[v].add(u)
                new_diam = _bfs_diameter(current_adj, n)
                
                current_edges.append(e)
                dfs(i + 1, current_edges, current_adj, new_diam)
                current_edges.pop()
                
                # Undo
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                
                if time.time() > deadline:
                    return

        # Start BnB
        # We need a copy of base_adj
        start_adj = {i: set(base_adj[i]) for i in range(n)}
        start_diam = _bfs_diameter(start_adj, n)
        
        # To improve BnB, we can try different orderings or random restarts of the BnB
        # But standard BnB with good ordering is usually fine for small N.
        
        # Let's run DFS
        dfs(0, [], start_adj, start_diam)
        
        return best_solution

    sol = solve_bnb(candidates, base_adj, n, 4)
    
    if sol is None:
        return []
        
    # Final validation
    if _is_valid_solution(sol, n, base_edges):
        return [[int(u), int(v)] for u, v in sol]
    else:
        # Fallback
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        fallback = [[hub, v] for v in targets]
        if _is_valid_solution(fallback, n, base_edges):
            return [[int(u), int(v)] for u, v in fallback]
        return []

# EVOLVE-BLOCK-END
