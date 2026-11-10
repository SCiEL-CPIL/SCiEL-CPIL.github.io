import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial best: Two K3,3
    best_edges = []
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            best_edges.append([u, v])
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            best_edges.append([u, v])
    best_raw = 2
    
    A = list(range(6))
    B = list(range(6, 12))
    
    def get_cycle_lengths(adj):
        lengths = set()
        n = 12
        # We only care about lengths < best_raw? No, we need the set size.
        # But if we find a length not in {4,6} and we already have 4,6, raw becomes 3.
        # We can prune if len(lengths) >= best_raw and we are just adding more?
        # Actually, if we find a new length, raw increases.
        # We want to minimize raw.
        
        # Optimization: if we find 3 distinct lengths, we can stop if best_raw is 2?
        # No, we need to know if it's 1 or 2.
        # If we find 4 and 6, raw is at least 2.
        # If we find 4, 6, 8, raw is 3.
        # So if best_raw is 2, and we find 3 lengths, we can abort this graph.
        
        def dfs(node, path, visited):
            if time.time() > deadline:
                return False
            if len(path) >= 4 and path[0] == node:
                if min(path) == path[0]:
                    lengths.add(len(path))
                    if len(lengths) > best_raw:
                        return True # Abort
            if len(path) == 12:
                return False
            for neighbor in adj[node]:
                if neighbor == path[0] and len(path) >= 4:
                    if min(path) == path[0]:
                        lengths.add(len(path))
                        if len(lengths) > best_raw:
                            return True
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                if dfs(neighbor, path, visited):
                    path.pop()
                    visited.remove(neighbor)
                    return True
                path.pop()
                visited.remove(neighbor)
            return False
        
        for start in range(n):
            if time.time() > deadline:
                break
            if dfs(start, [start], {start}):
                break
                
        return lengths

    for _ in range(100000): # Max attempts, but limited by time
        if time.time() > deadline:
            break
            
        neighbors = [None] * 6
        valid = True
        for i in range(6):
            candidates = list(range(6, 12))
            random.shuffle(candidates)
            neighbors[i] = candidates[:3]
            
        b_deg = {v: 0 for v in B}
        for i in range(6):
            for v in neighbors[i]:
                b_deg[v] += 1
                
        if all(deg == 3 for deg in b_deg.values()):
            adj = [[] for _ in range(12)]
            for i in range(6):
                u = i
                for v in neighbors[i]:
                    adj[u].append(v)
                    adj[v].append(u)
                    
            lengths = get_cycle_lengths(adj)
            
            if lengths:
                raw = len(lengths)
                if raw < best_raw:
                    best_raw = raw
                    best_edges = [[u, v] for u in range(12) for v in adj[u] if u < v]
                    if best_raw == 1:
                        return best_edges
                        
    return best_edges
