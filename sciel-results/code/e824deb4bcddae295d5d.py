import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses Strassen's algorithm on three 2x2 blocks, achieving rank 21.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Rank is 21 (3 blocks * 7 Strassen steps)
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Define Strassen coefficients for 2x2 multiplication
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    v_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        v_indices.append([0 * 6 + k0, 0 * 6 + k1, 1 * 6 + k0, 1 * 6 + k1])
        
    w_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        w_indices.append([k0 * 2 + 0, k1 * 2 + 0, k0 * 2 + 1, k1 * 2 + 1])

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U[local_idx, r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    V[v_indices[g][local_idx], r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    W[w_indices[g][local_idx], r] = w_val

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0
                
    if not np.allclose(T_check, T_exp, atol=1e-10):
        rank_school = 24
        U_s = np.zeros((4, rank_school))
        V_s = np.zeros((12, rank_school))
        W_s = np.zeros((12, rank_school))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
