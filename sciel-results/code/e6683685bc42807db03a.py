import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    base_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                base_edges.add((i, j))
                
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
            
    def count_c4(adj):
        A2 = adj @ adj
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    c = A2[i, j]
                    if c >= 2:
                        count += c * (c - 1) // 2
        return count // 2

    # Initialize with polarity graph + one random edge
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in base_edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    non_edges = list(all_pairs - base_edges)
    random.shuffle(non_edges)
    
    best_c4 = float('inf')
    best_adj = None
    
    # Try a few random starting edges
    for start_idx in range(min(10, len(non_edges))):
        u, v = non_edges[start_idx]
        adj[u, v] = 1
        adj[v, u] = 1
        
        current_c4 = count_c4(adj)
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_adj = adj.copy()
            
        adj[u, v] = 0
        adj[v, u] = 0
        
        if time.time() > deadline - 30:
            break
            
    # Local search from best start
    if best_adj is not None:
        adj = best_adj.copy()
        current_c4 = best_c4
        current_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    current_edges.add((i, j))
                    
        temperature = 5.0
        cooling = 0.999
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 200
        
        for iteration in range(200000):
            if time.time() > deadline - 5:
                break
            if temperature < min_temp:
                break
            if no_improve > max_no_improve:
                temperature = 5.0
                no_improve = 0
                # Perturb
                for _ in range(3):
                    e_rm = random.choice(list(current_edges))
                    e_add = random.choice([e for e in all_pairs if e not in current_edges])
                    adj[e_rm[0], e_rm[1]] = 0
                    adj[e_rm[1], e_rm[0]] = 0
                    adj[e_add[0], e_add[1]] = 1
                    adj[e_add[1], e_add[0]] = 1
                    current_edges.remove(e_rm)
                    current_edges.add(e_add)
                current_c4 = count_c4(adj)
                continue
                
            # Random swap
            e_rm = random.choice(list(current_edges))
            non_edges_now = [e for e in all_pairs if e not in current_edges]
            if not non_edges_now:
                break
            e_add = random.choice(non_edges_now)
            
            adj[e_rm[0], e_rm[1]] = 0
            adj[e_rm[1], e_rm[0]] = 0
            adj[e_add[0], e_add[1]] = 1
            adj[e_add[1], e_add[0]] = 1
            
            new_c4 = count_c4(adj)
            delta = new_c4 - current_c4
            
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_edges.remove(e_rm)
                current_edges.add(e_add)
                current_c4 = new_c4
                no_improve = 0
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
            else:
                adj[e_rm[0], e_rm[1]] = 1
                adj[e_rm[1], e_rm[0]] = 1
                adj[e_add[0], e_add[1]] = 0
                adj[e_add[1], e_add[0]] = 0
                no_improve += 1
                
            temperature *= cooling
            
    # Extract result
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                edges.add((i, j))
                
    # Ensure 91 edges
    if len(edges) != 91:
        if len(edges) < 91:
            for e in all_pairs:
                if len(edges) >= 91: break
                if e not in edges: edges.add(e)
        else:
            el = list(edges)
            random.shuffle(el)
            edges = set(el[:91])
            
    return [[u, v] for u, v in edges]
