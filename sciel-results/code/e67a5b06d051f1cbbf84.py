import random
import time
import itertools

def construct():
    n = 18
    # Initialize with Turan graph K6,6,6
    G = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            part_i = 0 if i < 6 else (1 if i < 12 else 2)
            part_j = 0 if j < 6 else (1 if j < 12 else 2)
            if part_i != part_j:
                G[i][j] = 1
                G[j][i] = 1
    
    def is_k4_free(G):
        for u, v, w, x in itertools.combinations(range(n), 4):
            if G[u][v] and G[u][w] and G[u][x] and \
               G[v][w] and G[v][x] and \
               G[w][x]:
                return False
        return True

    def max_triangle_free_size(G):
        triangles = []
        for u, v, w in itertools.combinations(range(n), 3):
            if G[u][v] and G[v][w] and G[w][u]:
                mask = (1 << u) | (1 << v) | (1 << w)
                triangles.append(mask)
        
        if not triangles:
            return n
            
        for k in range(n, 0, -1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                valid = True
                for t in triangles:
                    if (mask & t) == t:
                        valid = False
                        break
                if valid:
                    return k
        return 0

    start_time = time.time()
    timeout = 130 # Safety margin
    
    best_graph = [row[:] for row in G]
    best_score = max_triangle_free_size(best_graph)
    
    current_graph = [row[:] for row in G]
    current_score = best_score
    
    temp = 1.0
    cool = 0.99
    
    while time.time() - start_time < timeout:
        u, v = random.sample(range(n), 2)
        
        current_graph[u][v] ^= 1
        current_graph[v][u] ^= 1
        
        if not is_k4_free(current_graph):
            current_graph[u][v] ^= 1
            current_graph[v][u] ^= 1
            continue
            
        new_score = max_triangle_free_size(current_graph)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_graph = [row[:] for row in current_graph]
        else:
            diff = new_score - current_score
            if diff > 0:
                if random.random() < pow(2.718, -diff / temp):
                    current_score = new_score
                else:
                    current_graph[u][v] ^= 1
                    current_graph[v][u] ^= 1
            
        temp *= cool
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i][j]:
                edges.append([i, j])
    return edges
