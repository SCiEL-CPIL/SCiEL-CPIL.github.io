import time
import random
import numpy as np

def count_c4s(adj_matrix):
    # adj_matrix is n x n boolean
    # Number of C4s is sum over i<j of binom(common_neighbors(i,j), 2)
    n = adj_matrix.shape[0]
    # A^2 gives common neighbors
    # (A^2)_{ij} is number of length 2 paths from i to j
    # For i != j, this is the number of common neighbors
    A2 = np.dot(adj_matrix, adj_matrix)
    # We only care about i < j
    # binom(k, 2) = k * (k-1) / 2
    # We sum over all i < j
    # Extract upper triangle
    iu = np.triu_indices(n, k=1)
    counts = A2[iu]
    # Calculate binom(count, 2)
    # counts are integers
    c4_count = np.sum(counts * (counts - 1) // 2)
    return c4_count

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    m = 91
    
    # Initialize with random graph
    # Use a seed for reproducibility? No, random is fine.
    # But we want to find a good one.
    
    # Let's try a few random restarts
    best_edges = None
    best_score = float('inf')
    
    for restart in range(5):
        # Generate random edges
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                edges.add((u, v))
        
        edges = list(edges)
        adj_matrix = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
            
        current_score = count_c4s(adj_matrix)
        
        # Local search
        no_improve_count = 0
        while time.time() < deadline - 5:
            if no_improve_count > 1000:
                break
            
            # Pick random edge to remove
            u, v = random.choice(edges)
            # Pick random non-edge to add
            # We need a list of non-edges
            # Generating non-edges list is O(n^2)
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj_matrix[i, j]:
                        non_edges.append((i, j))
            
            if not non_edges:
                break
                
            x, y = random.choice(non_edges)
            
            # Try swap
            # Remove (u,v)
            adj_matrix[u, v] = 0
            adj_matrix[v, u] = 0
            edges.remove((u, v))
            
            # Add (x,y)
            adj_matrix[x, y] = 1
            adj_matrix[y, x] = 1
            edges.append((x, y))
            
            new_score = count_c4s(adj_matrix)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
            else:
                # Revert
                adj_matrix[x, y] = 0
                adj_matrix[y, x] = 0
                edges.remove((x, y))
                
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1
                edges.append((u, v))
                
                no_improve_count += 1
                
        if current_score < best_score:
            best_score = current_score
            best_edges = edges.copy()
            
    # Convert to list of lists
    return [list(e) for e in best_edges]
