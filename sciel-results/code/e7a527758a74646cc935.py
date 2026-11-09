import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 15
    # Start with empty graph
    adj = set()
    for i in range(n):
        for j in range(i+1, n):
            adj.add((i, j))
    
    # We'll try to remove edges to satisfy the constraint, starting from complete graph
    # Actually, let's build up from empty to maximize edges
    # Better: start with complete, remove edges that violate
    
    # Let's use a different approach: start empty, add edges greedily
    # But we need to check all 3-edge paths, which is expensive
    
    # Let's try: start with complete graph, find violations, remove edges to fix
    # This might be more efficient since we start with many edges
    
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            edges.add((i, j))
    
    def get_degree(v, edge_set):
        return sum(1 for e in edge_set if v in e)
    
    def has_violation(edge_set):
        """Check if any 3-edge path s-u-v-t has deg(s) == deg(t)"""
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        
        # Build adjacency list
        adj_list = [[] for _ in range(n)]
        for u, v in edge_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Check all 3-edge paths: s-u-v-t where (s,u), (u,v), (v,t) are edges
        # and s != t
        for u in range(n):
            for v in adj_list[u]:
                for t in adj_list[v]:
                    if t == u:
                        continue
                    # Path is s-u-v-t, so we need s adjacent to u, s != v, s != t
                    # Actually, the path is s-u-v-t, meaning edges (s,u), (u,v), (v,t)
                    # So s is a neighbor of u, t is a neighbor of v, s != t
                    # And we need deg(s) != deg(t)
                    for s in adj_list[u]:
                        if s == v or s == t:
                            continue
                        if deg[s] == deg[t]:
                            return True, (s, u, v, t)
        return False, None
    
    # This is too slow. Let's use a smarter approach.
    # The constraint: for every path s-u-v-t (3 edges), deg(s) != deg(t)
    # This means: if deg(s) == deg(t), then there must be NO path of length 3 between s and t
    
    # Alternative: group vertices by degree. If two vertices have the same degree,
    # they must be at distance > 3 or distance 1 (not connected by a 3-edge path)
    
    # Let's try a constructive approach with careful degree assignment
    # Assign degrees first, then build graph
    
    # For n=15, let's try to make degrees as distinct as possible
    # But we need to maximize edges, so we want high degrees
    
    # Let's try: assign degrees d[0] < d[1] < ... < d[14]
    # Then ensure that vertices with same degree are far apart
    
    # Actually, let's just do a greedy removal from complete graph
    # But check violations efficiently
    
    # Let's try a different strategy: use the fact that if we can make all degrees distinct,
    # then the constraint is automatically satisfied (since no two vertices have same degree)
    
    # Can we have 15 distinct degrees in a graph on 15 vertices?
    # Degrees range from 0 to 14. We need 15 distinct values, so we need degrees 0,1,2,...,14
    # But a graph with degree sequence 0,1,2,...,14 is impossible (sum = 105, which is odd)
    # So we can't have all distinct degrees
    
    # What if we have at most 2 vertices with the same degree?
    # Then we just need to ensure those pairs are at distance > 3
    
    # Let's try to construct a graph with as many edges as possible where
    # vertices with equal degree are at distance >= 4
    
    # Start with complete graph (105 edges), remove edges to fix violations
    
    # Let's implement a local search: start with complete, find a violation, remove an edge
    # that breaks it, repeat until no violations
    
    best_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            best_edges.add((i, j))
    
    # We'll do iterative improvement
    current_edges = set(best_edges)
    
    def find_one_violation(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        
        adj_list = [[] for _ in range(n)]
        for u, v in edge_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # For each pair (s,t) with deg[s] == deg[t], check if there's a 3-edge path
        # A 3-edge path s-u-v-t exists if there exist u,v such that
        # (s,u) in edges, (u,v) in edges, (v,t) in edges, s != t
        
        for s in range(n):
            for t in range(s+1, n):
                if deg[s] != deg[t]:
                    continue
                # Check if there's a path of length 3 from s to t
                # s -> u -> v -> t
                found = False
                for u in adj_list[s]:
                    for v in adj_list[u]:
                        if v == s:
                            continue
                        if t in adj_list[v]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return (s, t)
        return None
    
    # Iteratively remove edges to fix violations
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        violation = find_one_violation(current_edges)
        if violation is None:
            break
        
        s, t = violation
        # Remove an edge on a 3-edge path between s and t
        # Find one such path
        deg = [0] * n
        for u, v in current_edges:
            deg[u] += 1
            deg[v] += 1
        
        adj_list = [[] for _ in range(n)]
        for u, v in current_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Find a 3-edge path s-u-v-t
        path_found = False
        for u in adj_list[s]:
            for v in adj_list[u]:
                if v == s:
                    continue
                if t in adj_list[v]:
                    # Remove edge (s,u) or (u,v) or (v,t)
                    # Try removing (s,u) first
                    e1 = (min(s,u), max(s,u))
                    e2 = (min(u,v), max(u,v))
                    e3 = (min(v,t), max(v,t))
                    
                    # Try each edge removal
                    for e in [e1, e2, e3]:
                        if e in current_edges:
                            new_edges = set(current_edges)
                            new_edges.discard(e)
                            # Check if this fixes the violation (might create new ones, but let's proceed)
                            current_edges = new_edges
                            path_found = True
                            break
                    if path_found:
                        break
            if path_found:
                break
        
        if not path_found:
            break
    
    # Convert to list of lists
    result = []
    for u, v in current_edges:
        result.append([int(u), int(v)])
    
    return result
