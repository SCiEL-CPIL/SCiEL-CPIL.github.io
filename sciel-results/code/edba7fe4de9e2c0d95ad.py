"""Optimized matrix multiplication tensor decomposition for 4x5 * 5x6."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _make_strassen_block(n, m, p, i0, j0, k0):
    """
    Creates the 7-rank Strassen decomposition for a 2x2x2 block
    starting at indices (i0, j0, k0).
    Returns U, V, W columns for this block.
    """
    U = np.zeros((n * m, 7), dtype=float)
    V = np.zeros((m * p, 7), dtype=float)
    W = np.zeros((p * n, 7), dtype=float)
    
    # Global indices for local 0,0; 0,1; 1,0; 1,1
    # i: i0, i0+1
    # j: j0, j0+1
    # k: k0, k0+1
    
    # Helper to get global row indices
    # U row: i * m + j
    # V row: j * p + k
    # W row: k * n + i
    
    # M1: (A00 + A11)(B00 + B11) -> C00, C11
    U[i0*m + j0, 0] = 1.0
    U[(i0+1)*m + (j0+1), 0] = 1.0
    V[j0*p + k0, 0] = 1.0
    V[(j0+1)*p + (k0+1), 0] = 1.0
    W[k0*n + i0, 0] = 1.0
    W[(k0+1)*n + (i0+1), 0] = 1.0
    
    # M2: (A10 + A11)B00 -> C01
    U[(i0+1)*m + j0, 1] = 1.0
    U[(i0+1)*m + (j0+1), 1] = 1.0
    V[j0*p + k0, 1] = 1.0
    W[k0*n + (i0+1), 1] = 1.0 # C10? No, C01 is (i0, k0+1)
    # Correction: C01 corresponds to i=i0, k=k0+1.
    # W index is k*n + i.
    # C01 -> k=k0+1, i=i0. Row: (k0+1)*n + i0.
    # My previous manual derivation said W row 1 is C01?
    # Local W rows:
    # 0: k0, i0 (C00)
    # 1: k0, i0+1 (C10) -- Wait. W[k*n+i].
    # Local k=0, i=0 -> C00.
    # Local k=0, i=1 -> C10.
    # Local k=1, i=0 -> C01.
    # Local k=1, i=1 -> C11.
    
    # Let's re-verify Strassen formulas with local indices:
    # C00 = M1 + M4 - M5 + M6
    # C01 = M2 + M5
    # C10 = M3 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # M2 contributes to C01.
    # C01 is local (i=0, k=1).
    # Global W row: (k0+1)*n + i0.
    W[(k0+1)*n + i0, 1] = 1.0
    
    # M3: A00(B10 - B11) -> C10
    U[i0*m + j0, 2] = 1.0
    V[(j0+1)*p + k0, 2] = 1.0
    V[(j0+1)*p + (k0+1), 2] = -1.0
    # C10 is local (i=1, k=0).
    # Global W row: k0*n + (i0+1).
    W[k0*n + (i0+1), 2] = 1.0
    
    # M4: A11(B01 - B00) -> C00, C10
    U[(i0+1)*m + (j0+1), 3] = 1.0
    V[j0*p + (k0+1), 3] = 1.0
    V[j0*p + k0, 3] = -1.0
    # C00 is local (i=0, k=0). Row: k0*n + i0.
    W[k0*n + i0, 3] = 1.0
    # C10 is local (i=1, k=0). Row: k0*n + (i0+1).
    W[k0*n + (i0+1), 3] = 1.0
    
    # M5: (A00 + A01)B11 -> C00, C01
    U[i0*m + j0, 4] = 1.0
    U[i0*m + (j0+1), 4] = 1.0
    V[(j0+1)*p + (k0+1), 4] = 1.0
    # C00 is local (i=0, k=0). Row: k0*n + i0.
    W[k0*n + i0, 4] = -1.0
    # C01 is local (i=0, k=1). Row: (k0+1)*n + i0.
    W[(k0+1)*n + i0, 4] = 1.0
    
    # M6: (A10 - A00)(B00 + B01) -> C00
    U[(i0+1)*m + j0, 5] = 1.0
    U[i0*m + j0, 5] = -1.0
    V[j0*p + k0, 5] = 1.0
    V[j0*p + (k0+1), 5] = 1.0
    # C00 is local (i=0, k=0). Row: k0*n + i0.
    W[k0*n + i0, 5] = 1.0
    
    # M7: (A01 - A11)(B10 + B11) -> C11
    U[i0*m + (j0+1), 6] = 1.0
    U[(i0+1)*m + (j0+1), 6] = -1.0
    V[(j0+1)*p + k0, 6] = 1.0
    V[(j0+1)*p + (k0+1), 6] = 1.0
    # C11 is local (i=1, k=1). Row: (k0+1)*n + (i0+1).
    W[(k0+1)*n + (i0+1), 6] = 1.0
    
    return U, V, W

def _make_schoolbook_block(n, m, p, i_list, j_list, k_list):
    """
    Creates schoolbook decomposition for a set of indices.
    """
    rank = len(i_list) * len(j_list) * len(k_list)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in i_list:
        for j in j_list:
            for k in k_list:
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify the decomposition.
    """
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    
    U_mat = U.reshape(n, m, -1)
    V_mat = V.reshape(m, p, -1)
    W_mat = W.reshape(p, n, -1)
    
    # Check T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    # And cross terms == 0
    # Efficient check:
    # For each j, compute S_j[i,k,i',k'] = sum_r U[i,j,r] V[j,k,r] W[k',i',r]
    # This should be delta_{i,i'} delta_{k,k'}
    
    for j in range(m):
        # A_j[i,k,r] = U[i,j,r] * V[j,k,r]
        A_j = U_mat[:, j, :] * V_mat[j, :, :] # (n, p, r)
        # S_j[i,k,i',k'] = sum_r A_j[i,k,r] * W_mat[j, k', i', r]
        # W_mat[j] is (p, n, r) which is W[k', i', r]
        S_j = np.einsum('ikr,kir->ikip', A_j, W_mat[j], optimize=True)
        
        # Check S_j[i,k,i,k] == 1 and S_j[i,k,i',k'] == 0 for (i',k') != (i,k)
        for i in range(n):
            for k in range(p):
                if abs(S_j[i, k, i, k] - 1.0) > tol:
                    return False
                for ip in range(n):
                    for kp in range(p):
                        if ip == i and kp == k:
                            continue
                        if abs(S_j[i, k, ip, kp]) > tol:
                            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_list = []
    V_list = []
    W_list = []
    
    # Partitions
    # I: {0,1}, {2,3}
    # J: {0,1}, {2,3}, {4}
    # K: {0,1}, {2,3}, {4,5}
    
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3), (4, 4)] # (4,4) is size 1
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for i1, i2 in i_blocks:
        for j1, j2 in j_blocks:
            for k1, k2 in k_blocks:
                # Check if we can use Strassen (2x2x2)
                if i2 - i1 == 1 and j2 - j1 == 1 and k2 - k1 == 1:
                    U_b, V_b, W_b = _make_strassen_block(n, m, p, i1, j1, k1)
                    U_list.append(U_b)
                    V_list.append(V_b)
                    W_list.append(W_b)
                else:
                    # Schoolbook for this block
                    i_vals = list(range(i1, i2 + 1))
                    j_vals = list(range(j1, j2 + 1))
                    k_vals = list(range(k1, k2 + 1))
                    U_b, V_b, W_b = _make_schoolbook_block(n, m, p, i_vals, j_vals, k_vals)
                    U_list.append(U_b)
                    V_list.append(V_b)
                    W_list.append(W_b)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to full schoolbook if verification fails (should not happen)
        U_s, V_s, W_s = _make_schoolbook_block(n, m, p, list(range(n)), list(range(m)), list(range(p)))
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
