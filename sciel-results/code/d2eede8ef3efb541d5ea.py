import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency and edge set
    init_adj = [set() for _ in range(n)]
    init_edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        init_edge_set.add((u, v))
        init_adj[u].add(v)
        init_adj[v].add(u)

    def get_max_biclique(u, v, adj):
        # Orientation 1: u in L, v in R
        L1 = set(adj[v])
        R1 = set(adj[u])
        if u not in L1 or v not in R1:
            size1 = 0
        else:
            while True:
                new_L1 = {x for x in L1 if R1.issubset(adj[x])}
                new_R1 = {y for y in R1 if new_L1.issubset(adj[y])}
                if len(new_L1) == len(L1) and len(new_R1) == len(R1):
                    L1, R1 = new_L1, new_R1
                    break
                L1, R1 = new_L1, new_R1
            size1 = len(L1) * len(R1)

        # Orientation 2: u in R, v in L
        L2 = set(adj[u])
        R2 = set(adj[v])
        if v not in L2 or u not in R2:
            size2 = 0
        else:
            while True:
                new_L2 = {x for x in L2 if R2.issubset(adj[x])}
                new_R2 = {y for y in R2 if new_L2.issubset(adj[y])}
                if len(new_L2) == len(L2) and len(new_R2) == len(R2):
                    L2, R2 = new_L2, new_R2
                    break
                L2, R2 = new_L2, new_R2
            size2 = len(L2) * len(R2)

        if size1 >= size2:
            return size1, list(L1), list(R1)
        else:
            return size2, list(L2), list(R2)

    def solve_once():
        adj = [set(s) for s in init_adj]
        remaining = set(init_edge_set)
        bicliques = []
        
        while remaining:
            best_size = -1
            best_biclique = None
            
            # Shuffle to break ties and explore different paths
            edges_list = list(remaining)
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                size, L, R = get_max_biclique(u, v, adj)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
            
            if best_biclique is None:
                break
                
            L, R = best_biclique
            bicliques.append([L, R])
            
            # Remove edges from remaining and adj
            for x in L:
                for y in R:
                    e = (min(x, y), max(x, y))
                    if e in remaining:
                        remaining.remove(e)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
        return bicliques

    # Run multiple restarts
    best_sol = None
    best_count = float('inf')
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    while time.time() < deadline:
        sol = solve_once()
        if len(sol) < best_count:
            best_count = len(sol)
            best_sol = sol
            
        # Early exit if we found a perfect solution (e.g., 1 biclique, though unlikely)
        # Or just let it run to find better.
        
    if best_sol is None:
        # Fallback
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(P['n']) if any(a == u for a, v in P['edges'])]
        
    return best_sol
