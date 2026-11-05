"""Spectral-aware C4 minimization for 31 vertices, 91 edges."""
import numpy as np
from collections import defaultdict
import random
import time

def construct():
    n = 31
    m = 91
    
    start_time = time.time()
    timeout = 110  # leave 10s margin
    
    best_edges = None
    best_c4 = float('inf')
    
    for seed in range(8):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with a graph that has a spectrum close to the 
        # Hoffman-Singleton bound spectrum: one large eigenvalue, 
        # and the rest as small as possible (ideally 0 or -1)
        
        # Strategy: Start with a regular graph of degree 6 (since 2*91/31 ≈ 5.87)
        # Use a structured initialization based on quadratic residues or similar
        
        edges = set()
        
        # Create a base structure: use a circulant-like graph
        # For n=31, try steps that minimize C4 creation
        steps = [1, 2, 5, 7, 10, 11]  # 6 steps for degree 12, but we need 91 edges
        
        # Actually, let's build more carefully
        # Average degree is 5.87, so some vertices have degree 5, some 6
        # 91 = 31*5 + 16, so 16 vertices have degree 6, 15 have degree 5
        
        # Start with a 5-regular circulant: steps [1, 2, 3, 4, 5] would be degree 10
        # Let's use steps [1, 3, 7, 11, 18] for a 5-regular base (155 edges is too many)
        
        # Simpler: start with a random regular-ish graph
        # Use configuration model to create a 6-regular graph, then remove 14 edges
        
        # Create a 6-regular graph using configuration model
        half_edges = [i for i in range(n) for _ in range(6)]
        random.shuffle(half_edges)
        
        # Pair up stubs
        edges_set = set()
        used = [False] * len(half_edges)
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u != v:
                edge = (min(u, v), max(u, v))
                if edge not in edges_set:
                    edges_set.add(edge)
        
        # We might have fewer than 93 edges (n*6/2 = 93) due to duplicates
        # Add random edges until we have 93
        while len(edges_set) < n * 6 // 2:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edge = (min(u, v), max(u, v))
                if edge not in edges_set:
                    edges_set.add(edge)
        
        # Now remove 2 edges to get from 93 to 91
        edge_list = list(edges_set)
        random.shuffle(edge_list)
        # Remove edges that create the most C4s
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges_set:
            A[u, v] = 1
            A[v, u] = 1
        
        A2 = A @ A
        # Score each edge by how many C4s it participates in
        edge_scores = {}
        for (u, v) in edges_set:
            score = 0
            # C4s containing edge (u,v): for each common neighbor of u and v, 
            # plus paths of length 2 from u to v through other vertices
            # Actually, count pairs (x,y) where x~u, y~v, x~y, x!=v, y!=u
            for x in range(n):
                if x != v and A[u, x] == 1:
                    for y in range(n):
                        if y != u and A[v, y] == 1 and x != y and A[x, y] == 1:
                            score += 1
            edge_scores[(u, v)] = score
        
        # Remove the 2 edges with highest scores
        sorted_edges = sorted(edges_set, key=lambda e: edge_scores[e], reverse=True)
        for e in sorted_edges[:2]:
            edges_set.remove(e)
        
        edges = edges_set
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Compute initial C4 count
        A2 = A @ A
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = A2[i, j]
                if c >= 2:
                    c4_count += c * (c - 1) // 2
        
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = set(edges)
        
        # Spectral-aware local search with edge swaps
        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            
            # Recompute A2
            A2 = A @ A
            
            # Get current eigenvalues for spectral guidance
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.sort(eigenvalues)[::-1]
            # The C4 count relates to trace(A^4) = sum lambda_i^4
            # We want to minimize this while keeping the largest eigenvalue as large as possible
            
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            # Precompute non-edges
            non_edges = set()
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edges:
                        non_edges.add((i, j))
            
            non_edge_list = list(non_edges)
            
            for (a, b) in edge_list:
                if (time.time() - start_time) > timeout:
                    break
                if not non_edge_list:
                    break
                
                # Compute delta C4 for removing (a,b)
                delta_c4_rem = 0
                for x in range(n):
                    if x != a and A[b, x] == 1:
                        old_c = A2[b, x]
                        if old_c >= 1:
                            delta_c4_rem -= (old_c - 1)
                for y in range(n):
                    if y != b and A[a, y] == 1:
                        old_c = A2[a, y]
                        if old_c >= 1:
                            delta_c4_rem -= (old_c - 1)
                
                # Try random non-edges for adding, prioritize those that 
                # connect vertices with few common neighbors
                random.shuffle(non_edge_list)
                
                found_swap = False
                for (c, d) in non_edge_list[:30]:
                    if (c, d) == (a, b) or (c, d) == (b, a):
                        continue
                    
                    # Compute delta C4 for adding (c,d)
                    delta_c4_add = 0
                    for x in range(n):
                        if x != c and A[d, x] == 1:
                            delta_c4_add += A2[c, x]
                    for y in range(n):
                        if y != d and A[c, y] == 1:
                            delta_c4_add += A2[d, y]
                    
                    total_delta = delta_c4_rem + delta_c4_add
                    
                    if total_delta < 0:
                        # Perform the swap
                        edges.remove((a, b))
                        edges.add((c, d))
                        non_edge_list.remove((c, d))
                        non_edge_list.append((a, b))
                        
                        # Update A
                        A[a, b] = 0
                        A[b, a] = 0
                        A[c, d] = 1
                        A[d, c] = 1
                        
                        c4_count += total_delta
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    continue
        
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = set(edges)
        
        if (time.time() - start_time) > timeout:
            break
    
    # Convert best_edges to list of lists
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
