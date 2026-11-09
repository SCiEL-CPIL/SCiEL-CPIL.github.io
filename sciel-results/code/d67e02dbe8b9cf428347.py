import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s
    
    n = 15
    # Start with K_{7,8}: parts A (0-6) and B (7-14)
    A = list(range(7))
    B = list(range(7, 15))
    
    # Build initial edge set: all cross edges
    edges = set()
    for a in A:
        for b in B:
            edges.add((a, b) if a < b else (b, a))
    
    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set):
        deg = get_degrees(edge_set)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all 3-edge paths s-u-v-t
        for u in range(n):
            for v in adj[u]:
                if u > v:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    # Greedy: try to add edges
    def greedy_add(edge_set):
        deg = get_degrees(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try all possible edges not yet present
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edge_set:
                    continue
                # Tentatively add
                new_set = edge_set | {(u, v)}
                if is_valid(new_set):
                    edge_set = new_set
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
        return edge_set
    
    # Local search: remove an edge, then greedily re-add
    def local_search(edge_set, max_iter=100):
        best = edge_set
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            # Try removing a random edge
            edge_list = list(best)
            if not edge_list:
                break
            remove_edge = random.choice(edge_list)
            candidate = best - {remove_edge}
            # Greedily re-add
            candidate = greedy_add(candidate)
            if len(candidate) > len(best):
                best = candidate
        return best
    
    # Run greedy
    edges = greedy_add(edges)
    
    # Run local search
    edges = local_search(edges)
    
    # Convert to list of [u, v] with u < v
    result = []
    for u, v in edges:
        if u < v:
            result.append([u, v])
        else:
            result.append([v, u])
    
    return result
