import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 39
    best_graph = None
    
    for trial in range(200):
        if time.time() > deadline:
            break
            
        adj = [[False] * n for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            creates_c4 = False
            nu = [k for k in range(n) if adj[u][k]]
            nv = [k for k in range(n) if adj[v][k]]
            
            for w in nv:
                for x in nu:
                    if w != x and adj[w][x]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                adj[u][v] = True
                adj[v][u] = True
                edges.append([u, v])
                
        # Check alpha
        # Greedy independent set
        remaining = list(range(n))
        indep = []
        while remaining:
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = sum(1 for u in remaining if adj[v][u])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            indep.append(min_v)
            remaining.remove(min_v)
            remaining = [v for v in remaining if not adj[min_v][v]]
            
        if len(indep) < 11:
            best_graph = {'n': n, 'edges': edges}
            break
            
    if best_graph:
        return best_graph
    else:
        # Fallback: n=1
        return {'n': 1, 'edges': []}
