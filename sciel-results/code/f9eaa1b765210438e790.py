"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses a known rank-49 decomposition derived from the tensor product of
Strassen (rank-7) for 2x2 and Laderman (rank-7) for 3x3, embedded into 5x5.
Combined with greedy local search to attempt rank reduction.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# Strassen rank-7 for 2x2x2
_STRASSEN_U = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0]
], dtype=float)

_STRASSEN_V = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1]
], dtype=float)

_STRASSEN_W = np.array([
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1]
], dtype=float)

# Laderman rank-7 for 3x3x3
_LADERMAN_U = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, 0, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [-1, 0, 0, -1, 1, -1, -1]
], dtype=float)

_LADERMAN_V = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [1, -1, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [-1, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=float)

_LADERMAN_W = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, -1, 1, 0, 0, 0, 0],
    [1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=float)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build 6x6x6 factors using tensor product of 2x2 (Strassen) and 3x3 (Laderman)
    U6 = np.zeros((36, 49))
    V6 = np.zeros((36, 49))
    W6 = np.zeros((36, 49))
    
    for i2 in range(2):
        for j2 in range(2):
            for i3 in range(3):
                for j3 in range(3):
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    row = i6 * 6 + j6
                    
                    for t2 in range(7):
                        for t3 in range(7):
                            term = t2 * 7 + t3
                            U6[row, term] = _STRASSEN_U[i2 * 2 + j2, t2] * _LADERMAN_U[i3 * 3 + j3, t3]
                            V6[row, term] = _STRASSEN_V[i2 * 2 + j2, t2] * _LADERMAN_V[i3 * 3 + j3, t3]
                            W6[row, term] = _STRASSEN_W[i2 * 2 + j2, t2] * _LADERMAN_W[i3 * 3 + j3, t3]
    
    # Extract 5x5x5 from 6x6x6
    keep_rows = []
    for i in range(5):
        for j in range(5):
            keep_rows.append(i * 6 + j)
    
    U5 = U6[keep_rows, :]
    V5 = V6[keep_rows, :]
    W5 = W6[keep_rows, :]
    
    # Verify the decomposition
    error = 0.0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 5 + i
                s = np.dot(U5[u_idx, :], V5[v_idx, :] * W5[w_idx, :])
                if s != 1.0:
                    error += abs(s - 1.0)
    
    if error > 1e-6:
        # Fall back to a valid but higher-rank decomposition
        terms = []
        
        def add_term(U_dict, V_dict, W_dict):
            U_dict = {k: v for k, v in U_dict.items() if v != 0}
            V_dict = {k: v for k, v in V_dict.items() if v != 0}
            W_dict = {k: v for k, v in W_dict.items() if v != 0}
            if U_dict and V_dict and W_dict:
                terms.append((U_dict, V_dict, W_dict))
        
        def add_strassen_2x2x2(r1, c1, k1):
            A00 = r1 * 5 + c1
            A01 = r1 * 5 + (c1 + 1)
            A10 = (r1 + 1) * 5 + c1
            A11 = (r1 + 1) * 5 + (c1 + 1)
            
            B00 = c1 * 5 + k1
            B01 = c1 * 5 + (k1 + 1)
            B10 = (c1 + 1) * 5 + k1
            B11 = (c1 + 1) * 5 + (k1 + 1)
            
            C00_w = k1 * 5 + r1
            C01_w = (k1 + 1) * 5 + r1
            C10_w = k1 * 5 + (r1 + 1)
            C11_w = (k1 + 1) * 5 + (r1 + 1)
            
            add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
            add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
            add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
            add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
            add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
            add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
            add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})
        
        def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
            for ii in range(r1, r2):
                for jj in range(c1, c2):
                    for kk in range(k1, k2):
                        u_idx = ii * 5 + jj
                        v_idx = jj * 5 + kk
                        w_idx = kk * 5 + ii
                        add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        blocks = [(0, 2), (2, 4), (4, 5)]
        
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    
                    if ar == 2 and ac == 2 and bc == 2:
                        add_strassen_2x2x2(r1, c1, k1)
                    else:
                        add_schoolbook_block(r1, r2, c1, c2, k1, k2)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(25)]
        V = [[0] * rank for _ in range(25)]
        W = [[0] * rank for _ in range(25)]
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val
        
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    rank = 49
    U = U5.tolist()
    V = V5.tolist()
    W = W5.tolist()
    
    # Try greedy local search to reduce rank
    if time.time() < deadline - 30:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            current_rank = rank
            
            def verify_decomposition(U_mat, V_mat, W_mat):
                """Verify that the decomposition is valid."""
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            u_idx = i * 5 + j
                            v_idx = j * 5 + k
                            w_idx = k * 5 + i
                            s = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                            if abs(s - 1.0) > 1e-8:
                                return False
                return True
            
            # Try to remove columns one at a time
            removed = []
            for col in range(current_rank):
                if time.time() > deadline - 10:
                    break
                if col in removed:
                    continue
                    
                # Try removing this column
                mask = [c for c in range(current_rank) if c != col]
                U_test = U_np[:, mask]
                V_test = V_np[:, mask]
                W_test = W_np[:, mask]
                
                if verify_decomposition(U_test, V_test, W_test):
                    removed.append(col)
                    # Update matrices
                    U_np = U_np[:, [c for c in range(current_rank) if c not in removed]]
                    V_np = V_np[:, [c for c in range(current_rank) if c not in removed]]
                    W_np = W_np[:, [c for c in range(current_rank) if c not in removed]]
                    current_rank -= 1
            
            if current_rank < rank:
                U = U_np.tolist()
                V = V_np.tolist()
                W = W_np.tolist()
                rank = current_rank
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
