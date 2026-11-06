import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Safety margin for 120s limit
    
    best_result = {'n': 39, 'edges': []}
    
    for n in [46, 45, 44, 43, 42]:
        min_deg_req = n - 39
        
        # Try multiple random restarts with improved greedy
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use numpy for faster C4 checking
            # adj_matrix[i][j] = 1 if edge (i,j) exists
            adj_matrix = np.zeros((n, n), dtype=np.int8)
            
            # Generate random edge order, but prioritize edges that help balance degrees
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            # Greedy add edges with faster C4 check
            for u, v in edges:
                if time.time() > deadline:
                    break
                
                # Check if adding (u,v) creates a C4
                # A C4 u-v-w-x-u exists if there's w in N(u) and x in N(v) with edge (w,x)
                # Equivalently: for each w in N(u), check if N(w) intersects N(v)
                # Faster: check if sum over w in N(u) of (N(w) & N(v)) > 0
                
                creates_c4 = False
                # Get neighbors of u and v as boolean arrays
                nu = adj_matrix[u]
                nv = adj_matrix[v]
                
                # For each w that is neighbor of u, check if w has any neighbor in N(v)
                # i.e., check if row w of adj_matrix has any 1 in positions where nv is 1
                for w in range(n):
                    if nu[w]:
                        # Check if N(w) intersects N(v)
                        if np.dot(adj_matrix[w], nv) > 0:
                            creates_c4 = True
                            break
                
                if not creates_c4:
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
            
            # Check min degree
            degrees = adj_matrix.sum(axis=1)
            min_deg = int(np.min(degrees))
            if min_deg >= min_deg_req:
                # Construct edge list
                edge_list = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj_matrix[u, v]:
                            edge_list.append([u, v])
                best_result = {'n': n, 'edges': edge_list}
                return best_result
        
        if time.time() > deadline:
            break
    
    return best_result
