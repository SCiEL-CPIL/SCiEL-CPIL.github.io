import random
import time
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s kill

    def is_regular(subset, adj):
        k = len(subset)
        if k < 2:
            return True
        degrees = [0] * k
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i != j and adj[u, v] == 1:
                    degrees[i] += 1
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def count_bad_size6(adj, n):
        bads = []
        for subset in combinations(range(n), 6):
            if is_regular(subset, adj):
                bads.append(subset)
        return bads

    def verify_no_regular_induced(adj, n):
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                if is_regular(subset, adj):
                    return False
        return True

    def solve_for_n(n, time_limit):
        adj = np.random.randint(0, 2, (n, n))
        adj = (adj + adj.T) % 2
        np.fill_diagonal(adj, 0)
        
        best_adj = adj.copy()
        best_bad_count = float('inf')
        
        while time.time() < start_time + time_limit:
            bads = count_bad_size6(adj, n)
            current_bad = len(bads)
            
            if current_bad < best_bad_count:
                best_bad_count = current_bad
                best_adj = adj.copy()
                
            if current_bad == 0:
                if verify_no_regular_induced(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j] == 1:
                                edges.append([i, j])
                    return n, edges
                else:
                    adj = np.random.randint(0, 2, (n, n))
                    adj = (adj + adj.T) % 2
                    np.fill_diagonal(adj, 0)
                    continue
            
            if bads:
                bad_subset = random.choice(bads)
                u, v = random.sample(bad_subset, 2)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
        
        return None, None

    # Try n=21 first (ambitious)
    time_left = deadline - time.time()
    if time_left > 30:
        result_n, result_edges = solve_for_n(21, time_left - 5)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Try n=20 (known feasible)
    time_left = deadline - time.time()
    if time_left > 10:
        result_n, result_edges = solve_for_n(20, time_left - 5)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Try n=19
    time_left = deadline - time.time()
    if time_left > 5:
        result_n, result_edges = solve_for_n(19, time_left - 2)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Fallback
    return {'n': 5, 'edges': []}
