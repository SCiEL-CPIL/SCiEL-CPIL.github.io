import time
import numpy as np
import random

def construct():
    n = 40
    deadline = time.time() + 115.0
    
    # Initialize with 8 disjoint K5s
    A = np.zeros((n, n), dtype=np.int64)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                A[u, v] = 1
                A[v, u] = 1
                
    def compute_A5(mat):
        A2 = mat @ mat
        A3 = A2 @ mat
        A4 = A3 @ mat
        A5 = A4 @ mat
        return A5

    best_A = A.copy()
    best_count = np.sum(A) // 2
    
    def has_path_length_5_optimized(u, v, A):
        # Check for path u-x-y-z-w-v
        N_u = np.where(A[u, :] == 1)[0]
        N_v = np.where(A[v, :] == 1)[0]
        
        # Filter neighbors
        N_u = [x for x in N_u if x != v]
        N_v = [w for w in N_v if w != u]
        
        for x in N_u:
            for w in N_v:
                if x == w:
                    continue
                
                N_x = np.where(A[x, :] == 1)[0]
                N_w = np.where(A[w, :] == 1)[0]
                
                # Filter again to ensure simple path and exclude u, v
                N_x = [y for y in N_x if y != u and y != v and y != w and y != x]
                N_w = [z for z in N_w if z != u and z != v and z != x and z != w]
                
                for y in N_x:
                    for z in N_w:
                        if y == z:
                            continue
                        if A[y, z] == 1:
                            return True
        return False

    def can_add(u, v, A, A5):
        if A[u, v] == 1:
            return False
        if A5[u, v] == 0:
            return True
        return not has_path_length_5_optimized(u, v, A)

    def greedy_add(A, A5):
        count = np.sum(A) // 2
        improved = True
        while improved and time.time() < deadline:
            improved = False
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if can_add(u, v, A, A5):
                    A[u, v] = 1
                    A[v, u] = 1
                    A5 = compute_A5(A)
                    improved = True
                    current_count = np.sum(A) // 2
                    if current_count > count:
                        count = current_count
        return A, A5, count

    # Initial greedy
    A, A5, count = greedy_add(A, A5)
    if count > best_count:
        best_count = count
        best_A = A.copy()

    # Local search with kicks
    while time.time() < deadline:
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 1]
        if not edges:
            break
            
        rem_u, rem_v = random.choice(edges)
        
        A_kick = A.copy()
        A_kick[rem_u, rem_v] = 0
        A_kick[rem_v, rem_u] = 0
        A5_kick = compute_A5(A_kick)
        
        A_new, A5_new, count_new = greedy_add(A_kick, A5_kick)
        
        if count_new > best_count:
            best_count = count_new
            best_A = A_new.copy()
            A = A_new
            A5 = A5_new
        elif count_new > np.sum(A) // 2:
            A = A_new
            A5 = A5_new

    edges_out = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges_out.append([u, v])
    return edges_out
