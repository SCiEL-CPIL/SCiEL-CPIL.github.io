import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 2, 13
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    # 6 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4) = 46
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    def add_strassen_block(k0, col_offset):
        # Strassen 2x2x2 decomposition: 7 multiplications
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        # C00 = M1 - M4 - M3 + M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M6 + M2 - M5
        
        a_coeffs = [
            {0: 1.0, 3: 1.0},      # M1: A00 + A11
            {2: 1.0, 3: 1.0},      # M2: A10 + A11
            {0: 1.0},              # M3: A00
            {3: 1.0},              # M4: A11
            {0: 1.0, 1: 1.0},      # M5: A00 + A01
            {2: 1.0, 0: -1.0},     # M6: A10 - A00
            {1: 1.0, 3: -1.0},     # M7: A01 - A11
        ]
        
        b_coeffs = [
            {k0: 1.0, p + k0 + 1: 1.0},           # M1: B0k0 + B1k1
            {k0: 1.0},                             # M2: B0k0
            {k0 + 1: 1.0, p + k0 + 1: -1.0},      # M3: B0k1 - B1k1
            {p + k0: 1.0, k0: -1.0},              # M4: B1k0 - B0k0
            {p + k0 + 1: 1.0},                     # M5: B1k1
            {k0: 1.0, k0 + 1: 1.0},               # M6: B0k0 + B0k1
            {p + k0: 1.0, p + k0 + 1: 1.0},       # M7: B1k0 + B1k1
        ]
        
        c_combos = [
            (C_idx(0, k0),     [(0, 1.0), (3, -1.0), (2, -1.0), (6, 1.0)]),
            (C_idx(0, k0 + 1), [(4, 1.0), (2, 1.0)]),
            (C_idx(1, k0),     [(1, 1.0), (3, 1.0)]),
            (C_idx(1, k0 + 1), [(0, 1.0), (5, -1.0), (1, 1.0), (4, -1.0)]),
        ]
        
        for t in range(7):
            col = col_offset + t
            for a_row, val in a_coeffs[t].items():
                U[a_row, col] += val
            for b_row, val in b_coeffs[t].items():
                V[b_row, col] += val
        
        for c_row, terms in c_combos:
            for t, coeff in terms:
                W[c_row, col_offset + t] += coeff
    
    def add_single_block(k, col_offset):
        # Schoolbook 2x2x1: 4 multiplications
        # C0k = A00*B0k + A01*B1k
        # C1k = A10*B0k + A11*B1k
        prods = [
            (A_idx(0, 0), B_idx(0, k)),
            (A_idx(0, 1), B_idx(1, k)),
            (A_idx(1, 0), B_idx(0, k)),
            (A_idx(1, 1), B_idx(1, k)),
        ]
        for t, (a_row, b_row) in enumerate(prods):
            col = col_offset + t
            U[a_row, col] = 1.0
            V[b_row, col] = 1.0
        
        W[C_idx(0, k), col_offset + 0] += 1.0
        W[C_idx(0, k), col_offset + 1] += 1.0
        W[C_idx(1, k), col_offset + 2] += 1.0
        W[C_idx(1, k), col_offset + 3] += 1.0
    
    # Build the decomposition
    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    add_single_block(12, col)
    col += 4
    
    if time.time() > deadline:
        # Time pressure: return what we have (should be valid)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Vectorized verification
    target_a = []
    target_b = []
    target_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_a.append(i * m + j)
                target_b.append(j * p + k)
                target_c.append(k * n + i)
    
    target_a = np.array(target_a)
    target_b = np.array(target_b)
    target_c = np.array(target_c)
    
    U_t = U[target_a, :]
    V_t = V[target_b, :]
    W_t = W[target_c, :]
    sums = np.sum(U_t * V_t * W_t, axis=1)
    valid = np.all(np.abs(sums - 1.0) < 1e-9)
    
    if not valid:
        # Fallback to schoolbook rank 52
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
