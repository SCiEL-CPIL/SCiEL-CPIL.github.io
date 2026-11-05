import time
import random
import math
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
    
    def get_random_edge(adj):
        """Pick a random edge (u, v) with u < v."""
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and (adj[u] & (1 << v)):
                return u, v

    def get_random_non_edge(adj):
        """Pick a random non-edge (u, v) with u < v."""
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and not (adj[u] & (1 << v)):
                return u, v

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
                ru, rv = get_random_edge(adj)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                au, av = get_random_non_edge(adj)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
        
        current_score = score(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
        
        # SA loop
        T = 5.0
        T_min = 0.1
        steps_per_temp = 500
        
        while T > T_min and time_ok():
            for step in range(steps_per_temp):
                if not time_ok():
                    break
                
                # Pick a random edge to remove
                ru, rv = get_random_edge(adj)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                # Pick a random non-edge to add
                au, av = get_random_non_edge(adj)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                
                new_score = score(adj)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
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
            
            T *= 0.95
        
        # Polish Phase: Deterministic Local Search
        improved = True
        while improved and time_ok():
            improved = False
            # List current edges
            cur_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        cur_edges.append((u, v))
            
            # List non-edges
            non_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        non_edges.append((u, v))
            
            base_score = score(adj)
            found_improvement = False
            
            for e1 in cur_edges:
                if found_improvement or not time_ok(): break
                u1, v1 = e1
                # Remove e1
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                for e2 in non_edges:
                    u2, v2 = e2
                    # Add e2
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    s = score(adj)
                    
                    # Undo add e2
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    
                    if s < base_score:
                        # Accept swap: re-add e2, keep e1 removed
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        
                        base_score = s
                        improved = True
                        found_improvement = True
                        if s < best_score:
                            best_score = s
                            best_adj = adj[:]
                        break # Break inner loop
                
                # If we didn't find improvement for this e1, we must undo the removal of e1
                if not found_improvement:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
            
            # If we made an improvement, the lists are stale, so the while loop restarts

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
