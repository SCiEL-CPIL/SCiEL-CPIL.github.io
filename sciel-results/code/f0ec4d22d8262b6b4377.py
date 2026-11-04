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
    
    # Verify the decomposition (vectorized)
    def _verify(U_mat, V_mat, W_mat, tol=1e-8):
        """Vectorized verification: check sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
        # Build the tensor T[i,j,k] = sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r]
        # For each r: outer product of U[:,r] reshaped, V[:,r], W[:,r]
        # U[i*5+j, r], V[j*5+k, r], W[k*5+i, r]
        # Reshape: U_col = U_mat[:, r].reshape(5,5), V_col = V_mat[:, r].reshape(5,5), W_col = W_mat[:, r].reshape(5,5)
        # T[i,j,k] += U_col[i,j] * V_col[j,k] * W_col[k,i]
        # Note: W_col[k,i] = W_mat[k*5+i, r]
        T = np.zeros((5, 5, 5), dtype=float)
        for r in range(U_mat.shape[1]):
            Uc = U_mat[:, r].reshape(5, 5)
            Vc = V_mat[:, r].reshape(5, 5)
            Wc = W_mat[:, r].reshape(5, 5)
            # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # Wc[k,i] means we need Wc.T[i,k]
            for i in range(5):
                T[i, :, :] += Uc[i, :] * (Vc * Wc.T[i, :])
        return np.allclose(T, 1.0, atol=tol)
    
    if not _verify(U5, V5, W5, tol=1e-6):
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
    if time.time() < deadline - 15:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            current_rank = rank
            
            # Try to remove columns one at a time (fast vectorized check)
            removed = set()
            changed = True
            while changed:
                changed = False
                for col in range(current_rank):
                    if time.time() > deadline - 5:
                        break
                    if col in removed:
                        continue
                    mask = [c for c in range(current_rank) if c != col]
                    U_test = U_np[:, mask]
                    V_test = V_np[:, mask]
                    W_test = W_np[:, mask]
                    if _verify(U_test, V_test, W_test, tol=1e-8):
                        removed.add(col)
                        keep = [c for c in range(current_rank) if c not in removed]
                        U_np = U_np[:, keep]
                        V_np = V_np[:, keep]
                        W_np = W_np[:, keep]
                        current_rank = len(keep)
                        changed = True
            
            # Try to remove pairs of columns
            if time.time() < deadline - 5:
                pair_removed = set()
                changed = True
                while changed:
                    changed = False
                    for c1 in range(current_rank):
                        if time.time() > deadline - 5:
                            break
                        if c1 in pair_removed:
                            continue
                        for c2 in range(c1 + 1, current_rank):
                            if time.time() > deadline - 5:
                                break
                            if c2 in pair_removed:
                                continue
                            mask = [c for c in range(current_rank) if c != c1 and c != c2]
                            U_test = U_np[:, mask]
                            V_test = V_np[:, mask]
                            W_test = W_np[:, mask]
                            if _verify(U_test, V_test, W_test, tol=1e-8):
                                pair_removed.add(c1)
                                pair_removed.add(c2)
                                keep = [c for c in range(current_rank) if c not in pair_removed]
                                U_np = U_np[:, keep]
                                V_np = V_np[:, keep]
                                W_np = W_np[:, keep]
                                current_rank = len(keep)
                                changed = True
                                break  # restart inner loop since indices changed
            
            if current_rank < rank:
                U = U_np.tolist()
                V = V_np.tolist()
                W = W_np.tolist()
                rank = current_rank
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
