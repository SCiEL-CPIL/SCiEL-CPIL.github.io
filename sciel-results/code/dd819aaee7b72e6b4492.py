import time
import random
import sys
from collections import defaultdict

def _check_independent_set(adj, n, k):
    """
    Check if there is an independent set of size k.
    """
    deg = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    found = [False]
    
    def bt(idx, size, forbidden):
        if found[0]:
            return
        if size >= k:
            found[0] = True
            return
        if idx >= n:
            return
        if size + (n - idx) < k:
            return
            
        v = order[idx]
        
        if forbidden & (1 << v):
            bt(idx + 1, size, forbidden)
            return
        
        # Include v
        new_forbidden = forbidden | (1 << v) | adj[v]
        bt(idx + 1, size + 1, new_forbidden)
        
        if found[0]:
            return
        
        # Exclude v
        bt(idx + 1, size, forbidden)

    bt(0, 0, 0)
    return found[0]

def construct():
    deadline = time.time() + 140.0
    
    for n in range(43, 29, -1):
        if time.time() > deadline:
            break
            
        for trial in range(50):
            if time.time() > deadline:
                break
                
            adj = [0] * n
            common_counts = [[0] * n for _ in range(n)]
            
            # To add variety, we can randomize the tie-breaking in "best" selection
            # Or just use a fixed heuristic.
            
            while True:
                if time.time() > deadline:
                    break
                
                best_u = -1
                best_v = -1
                best_score = -1
                
                # Find best valid edge
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            continue
                        if common_counts[u][v] > 0:
                            continue
                        
                        deg_u = bin(adj[u]).count('1')
                        deg_v = bin(adj[v]).count('1')
                        score = deg_u + deg_v
                        # Add small random noise to break ties and avoid local minima
                        score += random.random() * 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_u = u
                            best_v = v
                
                if best_u == -1:
                    break
                
                u, v = best_u, best_v
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Update common_counts
                nu = adj[u]
                nv = adj[v]
                
                w = 0
                while nu:
                    if nu & 1:
                        if w != v:
                            common_counts[w][v] += 1
                            common_counts[v][w] += 1
                    nu >>= 1
                    w += 1
                
                w = 0
                while nv:
                    if nv & 1:
                        if w != u:
                            common_counts[u][w] += 1
                            common_counts[w][u] += 1
                    nv >>= 1
                    w += 1

            # Check independence
            if not _check_independent_set(adj, n, 11):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        
    # Fallback
    n_fallback = 30
    adj_fallback = [0] * n_fallback
    for i in range(10):
        u, v, w = 3 * i, 3 * i + 1, 3 * i + 2
        adj_fallback[u] |= (1 << v) | (1 << w)
        adj_fallback[v] |= (1 << u) | (1 << w)
        adj_fallback[w] |= (1 << u) | (1 << v)
    edges = []
    for u in range(n_fallback):
        for v in range(u + 1, n_fallback):
            if adj_fallback[u] & (1 << v):
                edges.append([u, v])
    return {'n': n_fallback, 'edges': edges}
