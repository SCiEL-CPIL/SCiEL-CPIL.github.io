import random
import time

def construct():
    # Set deadline
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s buffer
    
    n = 15
    best_edges = [[0, v] for v in range(1, n)]
    
    def get_degree_map(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid(edges):
        deg = get_degree_map(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check all paths s-u-v-t
        # s, u, v, t distinct
        for u in range(n):
            adj_u = adj[u]
            for v in adj_u:
                adj_v = adj[v]
                for s in adj_u:
                    if s == v:
                        continue
                    for t in adj_v:
                        if t == u or t == s:
                            continue
                        # Path s-u-v-t
                        if deg[s] == deg[t]:
                            return False
        return True

    # Initial check
    if not is_valid(best_edges):
        return best_edges

    best_score = len(best_edges)
    
    # Randomized Local Search
    current_edges = list(best_edges)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for _ in range(500000): # Iterations
        if time.time() > deadline:
            break
            
        # Try to add a random edge
        if random.random() < 0.7:
            # Find a missing edge
            present = set()
            for u, v in current_edges:
                present.add((min(u,v), max(u,v)))
            
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in present:
                        missing.append((i,j))
            
            if missing:
                u, v = random.choice(missing)
                new_edges = current_edges + [[u, v]]
                if is_valid(new_edges):
                    current_edges = new_edges
                    if len(current_edges) > best_score:
                        best_score = len(current_edges)
                        best_edges = current_edges
        
        # Try to swap
        else:
            if len(current_edges) < 2: continue
            i1 = random.randint(0, len(current_edges)-1)
            u1, v1 = current_edges[i1]
            
            present = set()
            for u, v in current_edges:
                present.add((min(u,v), max(u,v)))
            
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in present and (i,j) != (u1,v1) and (i,j) != (v1,u1):
                        missing.append((i,j))
            
            if missing:
                u2, v2 = random.choice(missing)
                new_edges = current_edges.copy()
                new_edges.pop(i1)
                new_edges.append([u2, v2])
                if is_valid(new_edges):
                    current_edges = new_edges
                    if len(current_edges) > best_score:
                        best_score = len(current_edges)
                        best_edges = current_edges

    # Final polish: try to add edges systematically
    for i in range(n):
        for j in range(i+1, n):
            if time.time() > deadline:
                break
            present = set()
            for u, v in current_edges:
                present.add((min(u,v), max(u,v)))
            if (i,j) not in present:
                new_edges = current_edges + [[i, j]]
                if is_valid(new_edges):
                    current_edges = new_edges
                    if len(current_edges) > best_score:
                        best_score = len(current_edges)
                        best_edges = current_edges
        if time.time() > deadline:
            break

    return best_edges
