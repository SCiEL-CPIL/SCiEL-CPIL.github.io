import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    # Start with K_{8,8} + 1 edge
    # Vertices 0-7 in part A, 8-15 in part B
    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    # Helper to get edge list from adjacency matrix
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    
    # Helper to compute score (max triangle degree sum)
    def compute_score(adj):
        deg = adj.sum(axis=1).astype(np.int32)
        # Triangle count matrix: T[i,j] = number of common neighbors of i and j
        # But we need max deg sum over triangles.
        # A triangle (u,v,w) exists if adj[u,v]=1, adj[v,w]=1, adj[w,u]=1.
        # Score = deg[u] + deg[v] + deg[w].
        
        # We can iterate over all triangles.
        # With n=16, max triangles is C(16,3) = 560.
        # We can use matrix multiplication to find common neighbors.
        # Common neighbors of u and v: dot(adj[u], adj[v])
        # But we need to check if they are connected to each other.
        
        max_sum = 0
        # Iterate over all triples u < v < w
        # To speed up, we can precompute something?
        # For n=16, 560 checks is very fast.
        
        # Let's just do the triple loop in Python for simplicity and correctness,
        # or vectorize if possible.
        # Vectorized approach:
        # Create a tensor? No, n is small.
        
        # Let's stick to a fast Python loop with numpy lookups?
        # Or just pure Python on the adjacency list?
        # Actually, the previous score function was pure Python.
        # Let's try to optimize the score computation.
        
        # Method:
        # For each edge (u, v), common neighbors are intersection of N(u) and N(v).
        # For each w in common neighbors, if u < v < w or u < w < v or w < u < v,
        # we have a triangle.
        
        # Let's just iterate u < v < w.
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[v, w] and adj[u, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    current_score = compute_score(adj)
    best_adj = adj.copy()
    best_score = current_score
    
    # Precompute all possible edges and non-edges
    # This helps in random selection
    
    start_time = time.time()
    timeout = 115.0
    T = 1.0
    cooling = 0.99995
    
    # List of all possible edges (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Current edge set for quick lookup and random choice
    current_edge_set = set()
    for u, v in all_edges:
        if adj[u, v]:
            current_edge_set.add((u, v))
    
    while time.time() - start_time < timeout:
        # Pick a random edge to remove
        if not current_edge_set:
            break
        remove_edge = random.choice(list(current_edge_set))
        
        # Pick a random non-edge to add
        # We need an edge NOT in current_edge_set
        # Instead of sampling until found, we can maintain a list of non-edges?
        # Or just sample. n=16, 120 total edges. 65 in graph. 55 non-edges.
        # Probability of hitting a non-edge is 55/120 ~ 0.45.
        # Expected 2.2 tries.
        
        add_edge = None
        for _ in range(20):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) not in current_edge_set:
                add_edge = (u, v)
                break
        
        if add_edge is None:
            continue
            
        if remove_edge == add_edge:
            continue

        u1, v1 = remove_edge
        u2, v2 = add_edge
        
        # Apply move to adjacency matrix
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        # Update edge set
        current_edge_set.discard(remove_edge)
        current_edge_set.add(add_edge)
        
        # Compute new score
        new_score = compute_score(adj)
        
        # Acceptance criteria
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
        else:
            delta = new_score - current_score
            if delta > 0:
                # Check for overflow in exp
                if delta / T > 700:
                    prob = 0.0
                else:
                    prob = math.exp(-delta / T)
                
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    current_edge_set.discard(add_edge)
                    current_edge_set.add(remove_edge)
            else:
                # Should not happen if new_score >= current_score and delta > 0 check above
                # But if new_score == current_score, delta = 0, we accept?
                # In SA, usually we accept equal moves to explore.
                # My logic: if new_score < current_score: accept.
                # else: delta = new - cur >= 0.
                # if delta > 0: probabilistic accept.
                # else (delta == 0): accept.
                pass # Already accepted implicitly by not reverting? 
                # Wait, in the 'else' branch, I didn't explicitly accept.
                # The variables adj and current_edge_set are already updated.
                # So if I don't revert, it's accepted.
                # So for delta == 0, it is accepted.
                # For delta > 0, I might revert.
                # This logic is correct.
        
        T *= cooling
    
    # Return best result
    best_edges = get_edges(best_adj)
    return {'n': n, 'edges': best_edges}
