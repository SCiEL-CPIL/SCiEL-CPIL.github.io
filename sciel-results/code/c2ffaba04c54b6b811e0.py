import numpy as np
import random
import time

def construct():
    N = 40
    E = 401
    
    # Initialize
    np.random.seed(42)
    A = np.zeros((N, N), dtype=np.int8)
    edges_set = set()
    while len(edges_set) < E:
        u = np.random.randint(N)
        v = np.random.randint(N)
        if u != v and (u, v) not in edges_set and (v, u) not in edges_set:
            edges_set.add((u, v))
            A[u, v] = 1
            A[v, u] = 1
            
    A2 = A @ A
    
    def get_max_book():
        # Max book size of existing edges
        # A2[i,j] is common neighbors.
        # We need max A2[i,j] where A[i,j] == 1
        # Use a mask
        return np.max(np.where(A == 1, A2, 0))

    max_book = get_max_book()
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    T = 2.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    best_A = A.copy()
    best_max_book = max_book
    
    # Precompute list of all potential edges for random selection
    # To pick a random non-edge, we can just sample and check.
    
    while time.time() < deadline:
        # Save state
        A_old = A.copy()
        A2_old = A2.copy()
        max_book_old = max_book
        
        # 1. Select edge to remove
        # Find edges with max book size
        A2_masked = np.where(A == 1, A2, 0)
        current_max = np.max(A2_masked)
        max_edges = np.argwhere(A2_masked == current_max)
        
        if len(max_edges) == 0:
            break
            
        # Pick one randomly
        idx = np.random.choice(len(max_edges))
        u, v = max_edges[idx]
        
        # Remove (u, v)
        A[u, v] = 0
        A[v, u] = 0
        
        # 2. Select edge to add
        # Pick random non-edge
        while True:
            x = np.random.randint(N)
            y = np.random.randint(N)
            if x != y and A[x, y] == 0:
                break
                
        A[x, y] = 1
        A[y, x] = 1
        
        # 3. Evaluate
        A2 = A @ A
        new_max_book = get_max_book()
        
        # 4. Accept/Reject
        if new_max_book < max_book:
            max_book = new_max_book
            if max_book < best_max_book:
                best_max_book = max_book
                best_A = A.copy()
        else:
            delta_E = new_max_book - max_book
            if delta_E == 0:
                # Accept neutral moves to explore
                max_book = new_max_book
            else:
                prob = np.exp(-delta_E / T)
                if np.random.rand() < prob:
                    max_book = new_max_book
                else:
                    # Revert
                    A = A_old
                    A2 = A2_old
                    max_book = max_book_old
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Extract edges from best_A
    result = []
    for i in range(N):
        for j in range(i+1, N):
            if best_A[i, j] == 1:
                result.append([i, j])
                
    return result
