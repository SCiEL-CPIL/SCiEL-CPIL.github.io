"""Optimized rank-minimal tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Strassen 2x2x2 decomposition
    # Local indices: il, jl, kl in {0,1}
    # U local: (il, jl) -> global: (i0+il)*m + (j0+jl)
    # V local: (jl, kl) -> global: (j0+jl)*p + (k0+kl)
    # W local: (kl, il) -> global: (k0+kl)*n + (i0+il)
    
    def get_strassen_block(i0, j0, k0):
        """
        Returns lists of (u_idx, u_val), (v_idx, v_val), (w_idx, w_val) for 7 Strassen terms.
        i0, j0, k0 are global offsets for the 2x2x2 block.
        """
        terms = []
        
        # Term 1: M1 = (a00+a11)(b00+b11)
        u1 = [((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v1 = [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w1 = [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (a10+a11)b00
        u2 = [((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v2 = [((j0+0)*p + (k0+0), 1)]
        w2 = [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = a00(b01-b11)
        u3 = [((i0+0)*m + (j0+0), 1)]
        v3 = [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)]
        w3 = [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = a11(b10-b00)
        u4 = [((i0+1)*m + (j0+1), 1)]
        v4 = [((j0+1)*p + (k0+0), 1), ((j0+0)*p + (k0+0), -1)]
        w4 = [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (a00+a01)b11
        u5 = [((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)]
        v5 = [((j0+1)*p + (k0+1), 1)]
        w5 = [((k0+0)*n + (i0+0), -1), ((k0+1)*n + (i0+0), 1)]
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (a10-a00)(b00+b01)
        u6 = [((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)]
        v6 = [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)]
        w6 = [((k0+1)*n + (i0+1), 1)]
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (a01-a11)(b10+b11)
        u7 = [((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v7 = [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w7 = [((k0+0)*n + (i0+0), 1)]
        terms.append((u7, w7))
        
        return terms

    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define the Strassen blocks
    # We have 3x4x5 = 3 * (4x5)
    # For each of the 3 rows of A, we can use Strassen for the 4x5 multiplication
    # 4x5 can be decomposed as two 2x2 blocks (covering j in {0,1} and j in {2,3}) 
    # but 4x5 is not a power of 2 square, so we need a more careful approach.
    
    # Actually, let's think about this differently.
    # We have A (3x4) * B (4x5) = C (3x5)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # The tensor has indices (i,j,k) with i in [0,3), j in [0,4), k in [0,5)
    # Total terms: 3*4*5 = 60
    
    # Strategy: Use Strassen for 2x2x2 blocks where possible, schoolbook for the rest.
    # We can partition:
    # - i in {0,1}, j in {0,1}, k in {0,1}: Strassen (7 terms)
    # - i in {0,1}, j in {0,1}, k in {2,3}: Strassen (7 terms)
    # - i in {0,1}, j in {0,1}, k in {4}: schoolbook (2 terms)
    # - i in {0,1}, j in {2,3}, k in {0,1}: Strassen (7 terms)
    # - i in {0,1}, j in {2,3}, k in {2,3}: Strassen (7 terms)
    # - i in {0,1}, j in {2,3}, k in {4}: schoolbook (2 terms)
    # - i in {2}, j in {0,1}, k in {0,1}: Strassen-like but only 1 row, so schoolbook (4 terms)
    # - i in {2}, j in {0,1}, k in {2,3}: schoolbook (4 terms)
    # - i in {2}, j in {0,1}, k in {4}: schoolbook (2 terms)
    # - i in {2}, j in {2,3}, k in {0,1}: schoolbook (4 terms)
    # - i in {2}, j in {2,3}, k in {2,3}: schoolbook (4 terms)
    # - i in {2}, j in {2,3}, k in {4}: schoolbook (2 terms)
    
    # Total: 7+7+2+7+7+2+4+4+2+4+4+2 = 52
    
    # Let's implement this:
    
    # Block 1: i0=0, j0=0, k0=0 (covers i in {0,1}, j in {0,1}, k in {0,1})
    terms1 = get_strassen_block(0, 0, 0)
    for u, v, w in terms1:
        U_col = np.zeros(dim_u)
        for idx, val in u:
            U_col[idx] = val
        V_col = np.zeros(dim_v)
        for idx, val in v:
            V_col[idx] = val
        W_col = np.zeros(dim_w)
        for idx, val in w:
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Block 2: i0=0, j0=0, k0=2 (covers i in {0,1}, j in {0,1}, k in {2,3})
    terms2 = get_strassen_block(0, 0, 2)
    for u, v, w in terms2:
        U_col = np.zeros(dim_u)
        for idx, val in u:
            U_col[idx] = val
        V_col = np.zeros(dim_v)
        for idx, val in v:
            V_col[idx] = val
        W_col = np.zeros(dim_w)
        for idx, val in w:
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Block 3: i0=0, j0=2, k0=0 (covers i in {0,1}, j in {2,3}, k in {0,1})
    terms3 = get_strassen_block(0, 2, 0)
    for u, v, w in terms3:
        U_col = np.zeros(dim_u)
        for idx, val in u:
            U_col[idx] = val
        V_col = np.zeros(dim_v)
        for idx, val in v:
            V_col[idx] = val
        W_col = np.zeros(dim_w)
        for idx, val in w:
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Block 4: i0=0, j0=2, k0=2 (covers i in {0,1}, j in {2,3}, k in {2,3})
    terms4 = get_strassen_block(0, 2, 2)
    for u, v, w in terms4:
        U_col = np.zeros(dim_u)
        for idx, val in u:
            U_col[idx] = val
        V_col = np.zeros(dim_v)
        for idx, val in v:
            V_col[idx] = val
        W_col = np.zeros(dim_w)
        for idx, val in w:
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Track which (i,j,k) tuples are covered by Strassen
    covered = set()
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            for k in [0, 1, 2, 3]:
                covered.add((i, j, k))
    
    # Schoolbook for all remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_col = np.zeros(dim_u)
                U_col[i*m + j] = 1
                V_col = np.zeros(dim_v)
                V_col[j*p + k] = 1
                W_col = np.zeros(dim_w)
                W_col[k*n + i] = 1
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)
    
    # Stack columns
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    rank = U.shape[1]
    
    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                s = float(np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback to full schoolbook if verification fails
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        rank = n*m*p
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
