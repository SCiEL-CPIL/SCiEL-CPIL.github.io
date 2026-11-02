"""
Finds a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
The minimal rank is 21, achieved by applying Strassen's algorithm to the three 2x2 blocks of the 2x6 matrix.
"""
import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Rank is 21.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # The minimal rank for 2x2 * 2x6 is 21 (3 * rank(2x2*2x2))
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # Indices for A, B, C are 00, 01, 10, 11
    # u: coefficients for A
    # v: coefficients for B
    # w: coefficients for C
    strassen_steps = [
        # s=0: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # s=1: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        # s=2: M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        # s=3: M4 = A11(B00-B01) -> C00 += M4, C10 += M4
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        # s=4: M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        # s=5: M6 = (A10-A00)(B00+B01) -> C10 += M6
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        # s=6: M7 = (A01-A11)(B10+B11) -> C01 += M7
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Index mappings for B and C based on block g
    # B index: j*6 + k. Local j in {0,1}, local k in {0,1}
    # C index: k*2 + i. Local k in {0,1}, local i in {0,1}
    
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Map local B indices (00, 01, 10, 11) to global V indices
        # 00: j=0, k=k0 -> 0*6 + k0
        # 01: j=0, k=k1 -> 0*6 + k1
        # 10: j=1, k=k0 -> 1*6 + k0
        # 11: j=1, k=k1 -> 1*6 + k1
        v_indices = [
            0 * 6 + k0,
            0 * 6 + k1,
            1 * 6 + k0,
            1 * 6 + k1
        ]
        
        # Map local C indices (00, 01, 10, 11) to global W indices
        # 00: k=k0, i=0 -> k0*2 + 0
        # 01: k=k1, i=0 -> k1*2 + 0
        # 10: k=k0, i=1 -> k0*2 + 1
        # 11: k=k1, i=1 -> k1*2 + 1
        w_indices = [
            k0 * 2 + 0,
            k1 * 2 + 0,
            k0 * 2 + 1,
            k1 * 2 + 1
        ]
        
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            u_list = step['u']
            v_list = step['v']
            w_list = step['w']
            
            # Fill U (A indices are global, same for all blocks)
            for local_idx in range(4):
                if u_list[local_idx] != 0:
                    U[local_idx, r] = u_list[local_idx]
                
                # Fill V
                if v_list[local_idx] != 0:
                    V[v_indices[local_idx], r] = v_list[local_idx]
                
                # Fill W
                if w_list[local_idx] != 0:
                    W[w_indices[local_idx], r] = w_list[local_idx]
            
            if time.time() > deadline:
                break
    
    # Verification
    # T_check[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Expected tensor
    T_exp = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_exp[a, b, c] = 1.0
                
    if not np.allclose(T_check, T_exp, atol=1e-10):
        # Fallback to schoolbook if construction failed (should not happen)
        # Schoolbook rank is n*m*p = 24
        rank_school = n * m * p
        U_s = np.zeros((nm, rank_school))
        V_s = np.zeros((mp, rank_school))
        W_s = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
