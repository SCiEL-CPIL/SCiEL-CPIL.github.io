import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n_target = 26
    num_edges = n_target * (n_target - 1) // 2
    
    # Precompute edge pairs
    edge_pairs = []
    for i in range(n_target):
        for j in range(i + 1, n_target):
            edge_pairs.append((i, j))
            
    # Initialize random coloring
    colors = np.random.randint(0, 5, size=num_edges)
    
    # Initialize Adjacency Matrices
    A = [np.zeros((n_target, n_target), dtype=np.int8) for _ in range(5)]
    deg = [np.zeros(n_target, dtype=np.int32) for _ in range(5)]
    
    # Fill matrices
    for idx, (i, j) in enumerate(edge_pairs):
        c = colors[idx]
        A[c][i, j] = 1
        A[c][j, i] = 1
        deg[c][i] += 1
        deg[c][j] += 1
        
    def get_A3_entry(M, u, v):
        col_v = M[:, v]
        V = M @ col_v
        return np.dot(M[u, :], V)
        
    def calc_energy():
        total_c4 = 0
        for c in range(5):
            A_c = A[c]
            A2 = A_c @ A_c
            A4 = A2 @ A2
            total_c4 += np.trace(A4) // 8
        return total_c4
        
    E = calc_energy()
    best_E = E
    best_colors = colors.copy()
    
    T = 1.0
    cooling_rate = 0.9999
    
    while time.time() < deadline:
        idx = random.randint(0, num_edges - 1)
        u, v = edge_pairs[idx]
        c_old = colors[idx]
        
        c_new = random.randint(0, 4)
        if c_new == c_old:
            continue
            
        A_old_mat = A[c_old]
        A_new_mat = A[c_new]
        
        val_new = get_A3_entry(A_new_mat, u, v)
        val_old = get_A3_entry(A_old_mat, u, v)
        
        delta_E = val_new - (val_old - deg[c_old][v])
        
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            colors[idx] = c_new
            
            A[c_old][u, v] = 0
            A[c_old][v, u] = 0
            A[c_new][u, v] = 1
            A[c_new][v, u] = 1
            
            deg[c_old][u] -= 1
            deg[c_old][v] -= 1
            deg[c_new][u] += 1
            deg[c_new][v] += 1
            
            E += delta_E
            if E < best_E:
                best_E = E
                best_colors = colors.copy()
                
            if E == 0:
                return {'n': n_target, 'colors': best_colors.tolist()}
                
        T *= cooling_rate
        
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
