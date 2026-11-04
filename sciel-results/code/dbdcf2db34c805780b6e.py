import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Rank-28 Strassen decomposition
    rank = 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Strassen 7-term decomposition for 2x2x2
    # For matrices A, B, C (2x2), C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen(col_offset, A_idx, B_idx, C_idx):
        """
        A_idx: [A00, A01, A10, A11] - indices in U
        B_idx: [B00, B01, B10, B11] - indices in V
        C_idx: [C00, C01, C10, C11] - indices in W
        """
        A00, A01, A10, A11 = A_idx
        B00, B01, B10, B11 = B_idx
        C00, C01, C10, C11 = C_idx
        
        # M1 = (A00+A11)(B00+B11)
        U[A00, col_offset] += 1
        U[A11, col_offset] += 1
        V[B00, col_offset] += 1
        V[B11, col_offset] += 1
        W[C00, col_offset] += 1
        W[C11, col_offset] += 1
        
        # M2 = (A10+A11)B00
        U[A10, col_offset+1] += 1
        U[A11, col_offset+1] += 1
        V[B00, col_offset+1] += 1
        W[C10, col_offset+1] += 1
        W[C11, col_offset+1] -= 1
        
        # M3 = A00(B10-B11)
        U[A00, col_offset+2] += 1
        V[B10, col_offset+2] += 1
        V[B11, col_offset+2] -= 1
        W[C01, col_offset+2] += 1
        W[C11, col_offset+2] += 1
        
        # M4 = A11(B01-B00)
        U[A11, col_offset+3] += 1
        V[B01, col_offset+3] += 1
        V[B00, col_offset+3] -= 1
        W[C00, col_offset+3] += 1
        W[C10, col_offset+3] += 1
        
        # M5 = (A00+A01)B11
        U[A00, col_offset+4] += 1
        U[A01, col_offset+4] += 1
        V[B11, col_offset+4] += 1
        W[C00, col_offset+4] -= 1
        W[C01, col_offset+4] += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[A10, col_offset+5] += 1
        U[A00, col_offset+5] -= 1
        V[B00, col_offset+5] += 1
        V[B10, col_offset+5] += 1
        W[C11, col_offset+5] += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[A01, col_offset+6] += 1
        U[A11, col_offset+6] -= 1
        V[B01, col_offset+6] += 1
        V[B11, col_offset+6] += 1
        W[C00, col_offset+6] += 1
    
    # Define the 4 blocks
    # Block 1: l in {0,1}, k in {0,1}
    # A indices: i*4+l for i in {0,1}, l in {0,1} -> [0,1,4,5]
    # B indices: l*4+k for l in {0,1}, k in {0,1} -> [0,1,4,5]
    # C indices: k*2+i for k in {0,1}, i in {0,1} -> [0,2,1,3]
    add_strassen(0, [0,1,4,5], [0,1,4,5], [0,2,1,3])
    
    # Block 2: l in {0,1}, k in {2,3}
    # A indices: [0,1,4,5]
    # B indices: l*4+k for l in {0,1}, k in {2,3} -> [2,3,6,7]
    # C indices: k*2+i for k in {2,3}, i in {0,1} -> [4,6,5,7]
    add_strassen(7, [0,1,4,5], [2,3,6,7], [4,6,5,7])
    
    # Block 3: l in {2,3}, k in {0,1}
    # A indices: i*4+l for i in {0,1}, l in {2,3} -> [2,3,6,7]
    # B indices: l*4+k for l in {2,3}, k in {0,1} -> [8,9,12,13]
    # C indices: k*2+i for k in {0,1}, i in {0,1} -> [0,2,1,3]
    add_strassen(14, [2,3,6,7], [8,9,12,13], [0,2,1,3])
    
    # Block 4: l in {2,3}, k in {2,3}
    # A indices: [2,3,6,7]
    # B indices: l*4+k for l in {2,3}, k in {2,3} -> [10,11,14,15]
    # C indices: k*2+i for k in {2,3}, i in {0,1} -> [4,6,5,7]
    add_strassen(21, [2,3,6,7], [10,11,14,15], [4,6,5,7])
    
    def verify(U_mat, V_mat, W_mat, r):
        S = np.zeros((nm, mp, pn), dtype=float)
        for t in range(r):
            S += np.einsum('a,b,c->abc', U_mat[:, t], V_mat[:, t], W_mat[:, t])
        return np.max(np.abs(S - T))
    
    max_err = verify(U, V, W, rank)
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = nm * mp * pn  # 32
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        idx = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    a = i * m + l
                    b = l * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to find linear dependencies
    if time.time() < deadline - 20:
        M = np.zeros((nm * mp * pn, rank), dtype=float)
        for r in range(rank):
            M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
        
        # SVD to find numerical rank
        try:
            U_svd, S_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10
            num_nonzero = np.sum(S_svd > tol)
            
            if num_nonzero < rank:
                # Find null space vectors
                null_vecs = Vt_svd[num_nonzero:]
                
                for nv in null_vecs:
                    if time.time() > deadline - 10:
                        break
                    # Try to eliminate a term using the null vector
                    for r_elim in range(rank):
                        if abs(nv[r_elim]) > 1e-8:
                            # Check if we can express term r_elim as a combination of others
                            other_cols = [s for s in range(rank) if s != r_elim]
                            M_sub = M[:, other_cols]
                            rhs = M[:, r_elim]
                            
                            try:
                                alpha, res, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                                residual = np.max(np.abs(M_sub @ alpha - rhs))
                                if residual < 1e-8:
                                    new_rank = rank - 1
                                    new_U = np.delete(U, r_elim, axis=1)
                                    new_V = np.delete(V, r_elim, axis=1)
                                    new_W = np.delete(W, r_elim, axis=1)
                                    
                                    err = verify(new_U, new_V, new_W, new_rank)
                                    if err < 1e-10:
                                        best_rank = new_rank
                                        best_U = new_U
                                        best_V = new_V
                                        best_W = new_W
                                        rank = new_rank
                                        U = new_U
                                        V = new_V
                                        W = new_W
                                        
                                        # Rebuild M
                                        M = np.zeros((nm * mp * pn, rank), dtype=float)
                                        for r in range(rank):
                                            M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
                                        break
                            except np.linalg.LinAlgError:
                                continue
        except Exception:
            pass
    
    # Convert to list with rounding
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
