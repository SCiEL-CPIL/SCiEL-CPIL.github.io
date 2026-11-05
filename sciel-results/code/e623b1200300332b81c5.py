import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 40
    
    def compute_A4(A):
        A2 = A @ A
        A4 = A2 @ A2
        return A4
    
    best_edges = []
    best_count = 0
    
    # Try multiple random restarts
    for seed in range(200):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random graph
        # Start slightly denser to force removals, or sparser to force additions?
        # Start with p=0.2
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.2:
                    A[i,j] = 1
                    A[j,i] = 1
        
        A4 = compute_A4(A)
        
        # Repair phase
        while True:
            if time.time() > deadline:
                break
                
            found_bad = False
            for u in range(n):
                for v in range(u+1, n):
                    if A[u,v] == 1 and A4[u,v] > 0:
                        # Remove edge
                        A[u,v] = 0
                        A[v,u] = 0
                        A4 = compute_A4(A)
                        found_bad = True
                        break
                if found_bad:
                    break
            if not found_bad:
                break
                
        # Improve phase
        while True:
            if time.time() > deadline:
                break
                
            found_add = False
            # Randomize the order of checking to avoid bias?
            # Or just check all.
            for u in range(n):
                for v in range(u+1, n):
                    if A[u,v] == 0 and A4[u,v] == 0:
                        # Add edge
                        A[u,v] = 1
                        A[v,u] = 1
                        A4 = compute_A4(A)
                        found_add = True
                        break
                if found_add:
                    break
            if not found_add:
                break
                
        # Count edges
        edge_count = np.sum(A) // 2
        
        if edge_count > best_count:
            best_count = edge_count
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i,j] == 1:
                        best_edges.append([i, j])
                        
    return best_edges
