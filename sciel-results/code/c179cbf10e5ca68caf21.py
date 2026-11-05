import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2
    
    def score(adj):
        """Return max degree sum over all triangles."""
        degs = [bin(adj[i]).count('1') for i in range(n)]
        best = 0
        # Find all triangles
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    # common neighbors of u and v
                    common = adj[u] & adj[v]
                    # remove u and v themselves
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # iterate over common neighbors w > v to avoid duplicates
                    w = v + 1
                    while w < n:
                        if common & (1 << w):
                            s = degs[u] + degs[v] + degs[w]
                            if s > best:
                                best = s
                        w += 1
        return best
    
    # Initial: K_{8,8} + edge (0,1)
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj = make_adj(edges)
    current_score = score(adj)
    
    best_score = current_score
    best_adj = adj[:]
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    # Simulated annealing with edge swaps
    random.seed(42)
    
    for restart in range(50):
        if not time_ok():
            break
        
        # For restarts > 0, start from best_adj with some perturbation
        if restart == 0:
            adj = make_adj(edges)
        else:
            adj = best_adj[:]
            # Perturb: do a few random swaps
            for _ in range(random.randint(5, 20)):
                # Remove a random edge
                all_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            all_edges.append((u, v))
                if not all_edges:
                    break
                ru, rv = random.choice(all_edges)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                # Add a random non-edge
                non_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                if not non_edges:
                    break
                au, av = random.choice(non_edges)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
        
        current_score = score(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
        
        # SA loop
        T = 5.0
        T_min = 0.1
        steps_per_temp = 200
        
        while T > T_min and time_ok():
            for step in range(steps_per_temp):
                if not time_ok():
                    break
                
                # Pick a random edge to remove
                all_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            all_edges.append((u, v))
                if not all_edges:
                    break
                
                ru, rv = random.choice(all_edges)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                # Pick a random non-edge to add
                non_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                if not non_edges:
                    # Undo
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    break
                
                au, av = random.choice(non_edges)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                
                new_score = score(adj)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj[:]
                else:
                    # Undo
                    adj[au] &= ~(1 << av)
                    adj[av] &= ~(1 << au)
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
            
            T *= 0.97
        
        if best_score <= 23:
            break
    
    # Convert best_adj to edge list
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
    
    # Verify
    assert len(final_edges) == target_edges, f"Expected {target_edges} edges, got {len(final_edges)}"
    
    return {'n': 16, 'edges': final_edges}
