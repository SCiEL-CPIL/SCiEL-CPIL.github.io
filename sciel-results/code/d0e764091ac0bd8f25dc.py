import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 20
    
    def is_regular(mask, adj):
        degs = []
        for v in range(n):
            if (mask >> v) & 1:
                degs.append((adj[v] & mask).bit_count())
        if not degs:
            return False
        d0 = degs[0]
        for d in degs[1:]:
            if d != d0:
                return False
        return True

    def find_violation(adj):
        # Check sizes 6 to n
        for k in range(6, n + 1):
            for comb in combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if is_regular(mask, adj):
                    return mask, comb
        return None, None

    def random_graph(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    for restart in range(500):
        if time.time() > deadline:
            break
            
        adj = random_graph(n)
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            mask, comb = find_violation(adj)
            if mask is None:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            if len(comb) >= 2:
                u, v = random.sample(comb, 2)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if random.random() < 0.05:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    return {'n': 5, 'edges': []}
