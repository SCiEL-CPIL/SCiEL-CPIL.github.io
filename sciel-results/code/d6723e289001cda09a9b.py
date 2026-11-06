import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Static solution: 8 disjoint K4s
    def get_static_graph():
        edges = []
        for b in range(8):
            start = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([start + i, start + j])
        return {'n': 32, 'edges': edges}

    def has_c5_exact(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for subset in itertools.combinations(range(n), 5):
            edge_count = 0
            degrees = [0] * 5
            for i in range(5):
                u = subset[i]
                for j in range(i + 1, 5):
                    v = subset[j]
                    if v in adj[u]:
                        edge_count += 1
                        degrees[i] += 1
                        degrees[j] += 1
            if edge_count == 5 and min(degrees) == 2:
                return True
        return False

    def has_independent_set_9_exact(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def backtrack(current_set, candidates):
            if len(current_set) == 9:
                return True
            if len(candidates) == 0:
                return False
            if len(current_set) + len(candidates) < 9:
                return False
            
            v = candidates[0]
            
            # Include v
            new_candidates = [u for u in candidates[1:] if u not in adj[v]]
            if backtrack(current_set + [v], new_candidates):
                return True
            
            # Exclude v
            if backtrack(current_set, candidates[1:]):
                return True
            
            return False
        
        return backtrack([], list(range(n)))

    def is_valid(n, edges):
        if has_c5_exact(n, edges):
            return False
        if has_independent_set_9_exact(n, edges):
            return False
        return True

    # Try to find better solutions
    # We try n=33, 34, 35
    # For each, we do a few random restarts
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
            
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Start with random graph
            edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < 0.1: # Low density
                        edges.append([u, v])
                        
            # Local search
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Check validity
                if is_valid(target_n, edges):
                    return {'n': target_n, 'edges': edges}
                
                # If not valid, try to fix
                # Heuristic: add random edge or remove random edge
                if random.random() < 0.5:
                    # Add edge
                    u = random.randint(0, target_n - 1)
                    v = random.randint(0, target_n - 1)
                    if u != v and [u, v] not in edges and [v, u] not in edges:
                        edges.append([u, v])
                else:
                    # Remove edge
                    if edges:
                        idx = random.randint(0, len(edges) - 1)
                        edges.pop(idx)
                        
    # Fallback
    return get_static_graph()
