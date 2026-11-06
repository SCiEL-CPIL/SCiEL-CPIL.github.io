import itertools
import time
import random

def construct():
    start_time = time.time()
    total_deadline = start_time + 140
    
    # Time budgets for each n
    # 22 is hardest, give it less time? Or more?
    # If 22 is possible, it should be found quickly if it exists.
    # If not, we waste time.
    # Let's give 22: 30s, 21: 30s, 20: 30s.
    
    budgets = {22: 30, 21: 30, 20: 30}
    
    for n in [22, 21, 20]:
        n_deadline = start_time + budgets[n]
        # But also respect total deadline
        n_deadline = min(n_deadline, total_deadline)
        
        for attempt in range(1000):
            if time.time() > n_deadline:
                break
                
            # Init random graph
            neighbors = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        neighbors[i] |= (1 << j)
                        neighbors[j] |= (1 << i)
                        
            # Repair
            for step in range(5000):
                if time.time() > n_deadline:
                    break
                    
                bad_S = find_one_bad_subgraph(neighbors, n)
                if bad_S is None:
                    # Success
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if neighbors[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Repair
                S_list = list(bad_S)
                u, v = random.sample(S_list, 2)
                if neighbors[u] & (1 << v):
                    neighbors[u] &= ~(1 << v)
                    neighbors[v] &= ~(1 << u)
                else:
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
                    
    # Fallback
    return {'n': 5, 'edges': []}

def find_one_bad_subgraph(neighbors, n):
    for k in range(6, n + 1):
        for S in itertools.combinations(range(n), k):
            S_list = list(S)
            mask = 0
            for v in S_list:
                mask |= (1 << v)
            
            # Check regularity
            degs = []
            for u in S_list:
                deg = (neighbors[u] & mask).bit_count()
                degs.append(deg)
            
            if len(set(degs)) == 1:
                return S
    return None
