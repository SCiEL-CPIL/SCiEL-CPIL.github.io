import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_graph = None
    best_score = float('inf')
    
    # Strategy: Start from a near-balanced random graph with 401 edges,
    # then use greedy local search that prioritizes breaking the largest books.
    # Key insight: An edge's book size = number of common neighbors.
    # We want to minimize the maximum book size across all edges.
    
    n = 40
    target_edges = 401
    
    # Phase 1: Generate diverse starting points using different constructions
    starting_graphs = []
    
    # Construction 1: Balanced bipartite + random remainder
    for k in [10, 11, 12, 13]:
        if time.time() > deadline:
            break
        n_s = k
        n_l = n - n_s
        e_sl = n_s * n_l
        e_l = target_edges - e_sl
        
        if 0 <= e_l <= n_l * (n_l - 1) // 2:
            # Create bipartite edges S-L
            edges = set()
            for u in range(n_s):
                for v in range(n_s, n):
                    edges.add((u, v))
            
            # Add random L-L edges
            all_l_edges = [(u, v) for u in range(n_s, n) for v in range(u+1, n)]
            random.shuffle(all_l_edges)
            for edge in all_l_edges[:e_l]:
                edges.add(edge)
            
            if len(edges) == target_edges:
                starting_graphs.append(list(edges))
    
    # Construction 2: Random near-regular graph
    for _ in range(3):
        if time.time() > deadline:
            break
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        starting_graphs.append(list(edges))
    
    # Phase 2: Local search with greedy book-breaking heuristic
    for start_edges in starting_graphs:
        if time.time() > deadline:
            break
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int8)
        edge_set = set()
        for u, v in start_edges:
            A[u, v] = 1
            A[v, u] = 1
            edge_set.add((min(u, v), max(u, v)))
        
        # Compute initial max book
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    common = np.dot(A[i], A[j])
                    if common > max_book:
                        max_book = common
        
        if max_book < best_score:
            best_score = max_book
            best_graph = list(edge_set)
        
        if best_score <= 7:
            return best_graph
        
        # Local search: try edge swaps to reduce max book
        current_max = max_book
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Find all edges with book size >= current_max - 1 (high-priority edges)
            high_book_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1:
                        common = np.dot(A[i], A[j])
                        if common >= current_max - 1:
                            high_book_edges.append((i, j, common))
            
            # Try to remove a high-book edge and add a low-book edge
            improved = False
            random.shuffle(high_book_edges)
            
            for u, v, book_uv in high_book_edges[:20]:  # Limit attempts per iteration
                if time.time() > deadline:
                    break
                
                # Try removing edge (u,v)
                A[u, v] = 0
                A[v, u] = 0
                edge_set.discard((min(u, v), max(u, v)))
                
                # Find a candidate edge to add that minimizes new max book
                best_add_score = float('inf')
                best_add_edge = None
                
                # Sample candidate edges to add
                candidates = []
                for _ in range(50):
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a != b:
                        cand = (min(a, b), max(a, b))
                        if cand not in edge_set:
                            candidates.append(cand)
                
                for a, b in candidates:
                    # Temporarily add edge (a,b)
                    A[a, b] = 1
                    A[b, a] = 1
                    
                    # Compute new max book (only need to check edges affected)
                    # Edges affected: those involving a or b, and the removed edge's neighbors
                    new_max = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            if A[i, j] == 1:
                                # Only compute if this edge could have high book
                                # Quick check: if either endpoint has high degree
                                deg_i = np.sum(A[i])
                                deg_j = np.sum(A[j])
                                if deg_i + deg_j < current_max - 2:
                                    continue
                                common = np.dot(A[i], A[j])
                                if common > new_max:
                                    new_max = common
                    
                    # Undo temporary addition
                    A[a, b] = 0
                    A[b, a] = 0
                    
                    if new_max < best_add_score:
                        best_add_score = new_max
                        best_add_edge = (a, b)
                
                if best_add_edge is not None and best_add_score < current_max:
                    # Accept the swap
                    a, b = best_add_edge
                    A[a, b] = 1
                    A[b, a] = 1
                    edge_set.add((a, b))
                    current_max = best_add_score
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Undo removal
                    A[u, v] = 1
                    A[v, u] = 1
                    edge_set.add((min(u, v), max(u, v)))
            
            if not improved:
                no_improve_count += 1
        
        if current_max < best_score:
            best_score = current_max
            best_graph = list(edge_set)
        
        if best_score <= 7:
            return best_graph
    
    return best_graph
