import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term: (u_dict, v_dict, w_dict) where keys are (row, col) tuples."""
    return [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

def _build_4x4x4_terms():
    """Build rank-49 terms for 4x4x4 matrix multiplication via 2-level Strassen."""
    strassen_2x2 = _strassen_2x2x2()
    
    # Top-level Strassen for 4x4x4 (treating 2x2 blocks)
    top_level = [
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1), ((1,1),-1)]),
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
        ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]),
    ]
    
    all_terms = []
    for t_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_level[t_idx]
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_2x2[s_idx]
            
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            
            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}
            
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))
    
    return all_terms

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1
                V[j*p + k][idx] = 1
                W[k*n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build rank-98 decomposition: two independent 4x4x4 Strassen decompositions
    terms_left = _build_4x4x4_terms()
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Shift k indices by 4 for the right half
    terms_right = []
    for u_dict, v_dict, w_dict in terms_left:
        u_new = u_dict
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))
    
    total_terms = terms_left + terms_right
    rank = len(total_terms)  # should be 98
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col_idx] = coeff
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p, rank):
            return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
