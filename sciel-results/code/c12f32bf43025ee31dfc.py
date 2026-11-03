import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = 140.0
    
    # Total Rank
    # 2 j-pairs * 3 k-pairs * 7 rank = 42
    # 1 j-single * 3 k-pairs * 1 rank = 3
    # Total = 45
    total_rank = 45
    
    dim_u = n * m # 10
    dim_v = m * p # 30
    dim_w = p * n # 12
    
    U = np.zeros((dim_u, total_rank), dtype=np.float64)
    V = np.zeros((dim_v, total_rank), dtype=np.float64)
    W = np.zeros((dim_w, total_rank), dtype=np.float64)
    
    r = 0
    
    # Define Strassen Patterns
    # Each pattern is a tuple of (U_dict, V_dict, W_dict)
    # Keys are (local_i, local_j), (local_j, local_k), (local_k, local_i)
    strassen_patterns = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process j-pairs (0,1) and (2,3)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            for u_pat, v_pat, w_pat in strassen_patterns:
                # Add to U
                for (li, lj), val in u_pat.items():
                    gi = li
                    gj = j0 + lj
                    idx_u = gi * m + gj
                    U[idx_u, r] += val
                
                # Add to V
                for (lj, lk), val in v_pat.items():
                    gj = j0 + lj
                    gk = k0 + lk
                    idx_v = gj * p + gk
                    V[idx_v, r] += val
                
                # Add to W
                for (lk, li), val in w_pat.items():
                    gk = k0 + lk
                    gi = li
                    idx_w = gk * n + gi
                    W[idx_w, r] += val
                
                r += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    # Process j=4
    j0 = 4
    for k0, k1 in k_pairs:
        # Rank 1 pattern
        # U: (0, j0), (1, j0) = 1
        U[0 * m + j0, r] = 1.0
        U[1 * m + j0, r] = 1.0
        
        # V: (j0, k0), (j0, k1) = 1
        V[j0 * p + k0, r] = 1.0
        V[j0 * p + k1, r] = 1.0
        
        # W: (k0, 0), (k0, 1), (k1, 0), (k1, 1) = 1
        W[k0 * n + 0, r] = 1.0
        W[k0 * n + 1, r] = 1.0
        W[k1 * n + 0, r] = 1.0
        W[k1 * n + 1, r] = 1.0
        
        r += 1
        if time.time() > deadline - 5:
            break

    # Verify r == total_rank
    if r != total_rank:
        # Should not happen, but fallback if it does
        pass

    # Construct target tensor T for verification
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Check error
    err = 0.0
    for c_idx in range(total_rank):
        contribution = np.einsum('r,s,t->rst', U[:, c_idx], V[:, c_idx], W[:, c_idx])
        err += np.sum((contribution - T)**2)
        
    if err > 1e-8:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=np.float64)
        V = np.zeros((dim_v, rank), dtype=np.float64)
        W = np.zeros((dim_w, rank), dtype=np.float64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
                    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
