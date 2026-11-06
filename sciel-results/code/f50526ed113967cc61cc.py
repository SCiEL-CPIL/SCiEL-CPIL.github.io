import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    def elapsed():
        return time.time() - start_time

    def is_regular(adj, subset):
        k = len(subset)
        if k < 6:
            return False
        degs = []
        for u in subset:
            d = 0
            for v in subset:
                if u != v and adj[u][v]:
                    d += 1
            degs.append(d)
        return len(set(degs)) == 1

    def find_one_regular_subset(adj, n):
        vertices = list(range(n))
        
        # Check size 6
        subsets_6 = list(itertools.combinations(vertices, 6))
        random.shuffle(subsets_6)
        for subset in subsets_6:
            if is_regular(adj, subset):
                return subset
        
        # Check size 7
        subsets_7 = list(itertools.combinations(vertices, 7))
        random.shuffle(subsets_7)
        for subset in subsets_7:
            if is_regular(adj, subset):
                return subset
                
        # Check larger
        for k in range(8, n+1):
            subsets_k = list(itertools.combinations(vertices, k))
            random.shuffle(subsets_k)
            for subset in subsets_k:
                if is_regular(adj, subset):
                    return subset
                    
        return None

    def local_search(adj, n, max_iters=200):
        for it in range(max_iters):
            if elapsed() > DEADLINE:
                return False
                
            S = find_one_regular_subset(adj, n)
            if S is None:
                return True
            
            pairs = [(u, v) for u in S for v in S if u < v]
            if not pairs:
                continue
            u, v = random.choice(pairs)
            adj[u][v] = not adj[u][v]
            adj[v][u] = not adj[u][v]
            
        return False

    best_n = 5
    best_edges = []
    
    for target_n in [22, 21, 20]:
        if elapsed() > DEADLINE - 10:
            break
            
        attempts = 5 if target_n == 22 else (10 if target_n == 21 else 20)
        
        found = False
        for attempt in range(attempts):
            if elapsed() > DEADLINE - 5:
                break
                
            adj = [[False] * target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            if local_search(adj, target_n, max_iters=200):
                edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                
                if find_one_regular_subset(adj, target_n) is None:
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
                    
        if found:
            break

    return {'n': best_n, 'edges': best_edges}
