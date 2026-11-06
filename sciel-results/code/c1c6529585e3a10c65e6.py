import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_graph = None
    best_score = float('inf')
    
    # Try different partition sizes k (size of small part S)
    # k=10 seemed promising (max book ~11)
    for k in range(8, 13):
        if time.time() > deadline:
            break
            
        n_s = k
        n_l = 40 - k
        
        # Edges between S and L are fixed: all of them
        # Number of edges in S-L: n_s * n_l
        e_sl = n_s * n_l
        
        # Edges needed in L:
        e_l = 401 - e_sl
        
        if e_l < 0 or e_l > n_l * (n_l - 1) // 2:
            continue
            
        # Try multiple random configurations for E_LL
        for _ in range(100):
            if time.time() > deadline:
                break
                
            # Randomly select e_l edges from L
            # Vertices in L are indexed 0 to n_l-1 locally
            # We will map them to global indices later
            
            # Generate all possible edges in L
            # To speed up, just pick random pairs
            edges_l = set()
            while len(edges_l) < e_l:
                u = np.random.randint(n_l)
                v = np.random.randint(n_l)
                if u != v:
                    edges_l.add((min(u, v), max(u, v)))
            
            # Build adjacency matrix
            A = np.zeros((40, 40), dtype=int)
            
            # Add S-L edges
            # S vertices: 0 to n_s-1
            # L vertices: n_s to 39
            for u in range(n_s):
                for v in range(n_s, 40):
                    A[u, v] = 1
                    A[v, u] = 1
            
            # Add L-L edges
            for u, v in edges_l:
                # Map local indices to global
                gu = u + n_s
                gv = v + n_s
                A[gu, gv] = 1
                A[gv, gu] = 1
            
            # Calculate max book size
            # Triangles containing edge (i,j) is sum(A[i,k]*A[j,k]) for k!=i,j
            # This is dot(A[i], A[j])
            # We only need to check edges where A[i,j]=1
            
            max_book = 0
            # Iterate over all pairs? 40*40 = 1600. Fast enough.
            for i in range(40):
                for j in range(i+1, 40):
                    if A[i, j] == 1:
                        # Count common neighbors
                        # A[i] dot A[j]
                        # Since A is symmetric and 0 diagonal, this is correct
                        common = np.dot(A[i], A[j])
                        if common > max_book:
                            max_book = common
            
            if max_book < best_score:
                best_score = max_book
                # Store the best graph
                best_edges = []
                for i in range(40):
                    for j in range(i+1, 40):
                        if A[i, j] == 1:
                            best_edges.append([i, j])
                best_graph = best_edges
                
            # Early stop if we hit the theoretical lower bound ~7
            if best_score <= 7:
                return best_graph

    return best_graph
