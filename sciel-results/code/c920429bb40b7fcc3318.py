"""Rewritten program for triangle-free diameter-four augmentation."""
import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    # Build adjacency for base path
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Precompute all candidate edges (non-adjacent pairs)
    candidates = []
    for u in range(n):
        for v in range(u + 2, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Convert to index for faster lookup
    cand_set = set(candidates)
    cand_idx = {c: i for i, c in enumerate(candidates)}
    m = len(candidates)
    
    # Precompute for each candidate edge, which other candidates it conflicts with (triangle creation)
    # Two edges (u,v) and (a,b) create a triangle if they share a vertex and the other endpoints are adjacent
    # Actually, for triangle-free: adding (u,v) is invalid if u and v have a common neighbor in the current graph.
    # Let's precompute conflicts: edge e1=(u,v) conflicts with e2=(a,b) if:
    #   - They share a vertex, say u=a, and (v,b) is in base or v-b is an edge we already have
    # This is complex. Let's just check triangle-free on the fly but efficiently.
    
    # For efficiency, let's use a different approach:
    # We'll use a branch-and-bound / backtracking search.
    # State: current set of added edges (as a frozenset for hashing)
    # We search for the minimum number of edges.
    
    # First, let's get a good upper bound via greedy.
    def greedy_solution():
        """Greedy: add edges that reduce far pairs the most."""
        adj = [set(base_adj[i]) for i in range(n)]
        
        def count_far():
            far = 0
            for i in range(n):
                dist = [-1] * n
                dist[i] = 0
                q = deque([i])
                while q:
                    node = q.popleft()
                    for nei in adj[node]:
                        if dist[nei] == -1:
                            dist[nei] = dist[node] + 1
                            q.append(nei)
                for j in range(i + 1, n):
                    if dist[j] > 4:
                        far += 1
            return far
        
        def is_tf():
            for i in range(n):
                for j in adj[i]:
                    if j <= i:
                        continue
                    if adj[i] & adj[j]:
                        return False
            return True
        
        far = count_far()
        added = []
        remaining = list(candidates)
        
        while far > 0:
            best = None
            best_far = far
            for e in remaining:
                u, v = e
                # Check if adding (u,v) keeps triangle-free
                if adj[u] & adj[v]:
                    continue
                # Temporarily add
                adj[u].add(v)
                adj[v].add(u)
                new_far = count_far()
                adj[u].discard(v)
                adj[v].discard(u)
                if new_far < best_far:
                    best_far = new_far
                    best = e
            if best is None:
                break
            u, v = best
            adj[u].add(v)
            adj[v].add(u)
            added.append(best)
            remaining.remove(best)
            far = best_far
        
        return added
    
    greedy_sol = greedy_solution()
    upper_bound = len(greedy_sol)
    
    # Now do exact search: try k = 1, 2, ..., upper_bound
    # For each k, check if there exists a valid set of k edges.
    
    def check_validity(added_edges):
        """Check if the set of added edges gives a valid solution."""
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                if adj[i] & adj[j]:
                    return False
        
        # Check diameter <= 4
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            if any(d > 4 for d in dist):
                return False
        
        return True
    
    # Branch and bound for exact minimum
    # We search over subsets of candidates of size k.
    # To make this efficient, we use the structure of the problem.
    
    # Let's try a smarter approach: since n=24 is small, and the answer is likely small (3-5),
    # we can enumerate combinations of candidates of size k for k=1,2,3,...
    
    # But C(276, 5) is too large. We need pruning.
    
    # Key insight: for diameter 4 on a path of 24, we need to "shortcut" the path.
    # The critical pairs are (0,23), (0,22), ..., etc.
    # Let's identify which pairs MUST be covered.
    
    # Actually, let's just do a well-optimized backtracking with strong pruning.
    
    deadline = time.time() + 80  # safety margin
    
    # Sort candidates by "importance" - edges that cover more far pairs first
    # Precompute for each candidate edge, how many far pairs it can potentially cover
    # This is approximate but helps ordering
    
    def edge_potential(u, v):
        """Estimate how many far pairs this edge helps with."""
        # An edge (u,v) helps with pairs (a,b) where the shortest path goes through u-v
        # Rough estimate: number of pairs (a,b) with a<=min(u,v) and b>=max(u,v)
        # that would otherwise have distance > 4
        lo, hi = min(u, v), max(u, v)
        count = 0
        for a in range(lo + 1):
            for b in range(hi, n):
                if b - a > 4:
                    count += 1
        return count
    
    # Precompute potentials
    potentials = [(edge_potential(u, v), (u, v)) for u, v in candidates]
    potentials.sort(key=lambda x: -x[0])
    ordered_candidates = [e for _, e in potentials]
    
    # Now do backtracking search
    # State: (index in ordered_candidates, current set of edges, current diameter info)
    
    # For efficiency, we'll maintain the adjacency and incrementally update distances
    # This is complex. Let's use a simpler approach with memoization on (frozenset of edges).
    
    # Actually, given the small expected answer (likely 3-5), let's just enumerate
    # combinations of size k with strong pruning.
    
    from itertools import combinations
    
    best_solution = greedy_sol
    
    for k in range(1, upper_bound + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Enumerate combinations of k edges from ordered_candidates
        # Use pruning: if partial set already creates a triangle, skip
        
        def search_combos(start, chosen, adj_copy):
            nonlocal found
            if found:
                return
            if time.time() > deadline:
                return
            
            if len(chosen) == k:
                # Check diameter
                # Build adj from base + chosen
                adj = [set(base_adj[i]) for i in range(n)]
                for u, v in chosen:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Check triangle-free
                for i in range(n):
                    for j in adj[i]:
                        if j <= i:
                            continue
                        if adj[i] & adj[j]:
                            return
                
                # Check diameter
                for i in range(n):
                    dist = [-1] * n
                    dist[i] = 0
                    q = deque([i])
                    while q:
                        node = q.popleft()
                        for nei in adj[node]:
                            if dist[nei] == -1:
                                dist[nei] = dist[node] + 1
                                q.append(nei)
                    if any(d > 4 for d in dist):
                        return
                
                found = True
                best_solution = list(chosen)
                return
            
            # Try adding each candidate from start onwards
            for i in range(start, m):
                if time.time() > deadline:
                    return
                e = ordered_candidates[i]
                u, v = e
                
                # Check if adding this edge creates a triangle with current chosen
                # u and v must not have a common neighbor in base + chosen
                common = set()
                for nei_u in base_adj[u]:
                    if nei_u in base_adj[v] or any(nei_u == a or nei_u == b for a, b in chosen):
                        common.add(nei_u)
                for nei_v in base_adj[v]:
                    if nei_v in base_adj[u] or any(nei_v == a or nei_v == b for a, b in chosen):
                        common.add(nei_v)
                
                # More precise check:
                adj_u = set(base_adj[u])
                adj_v = set(base_adj[v])
                for a, b in chosen:
                    if a == u:
                        adj_u.add(b)
                    elif b == u:
                        adj_u.add(a)
                    if a == v:
                        adj_v.add(b)
                    elif b == v:
                        adj_v.add(a)
                
                if adj_u & adj_v:
                    continue
                
                search_combos(i + 1, chosen + [e], None)
                if found:
                    return
        
        search_combos(0, [], None)
        
        if found:
            break
    
    # Validate final solution
    if not check_validity(best_solution):
        # Fallback to greedy
        best_solution = greedy_sol
    
    # Convert to proper format
    result = []
    for u, v in best_solution:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
