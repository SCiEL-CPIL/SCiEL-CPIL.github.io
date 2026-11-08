"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses Strassen-based block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# Precompute the Strassen 4x4*4x4 decomposition terms
# Each term is (u_dict, v_dict, w_dict) where keys are (row, col) indices
# For a 4x4*4x4 at base positions, we generate the 49 terms

def _build_strassen_4x4_terms():
    """Build the 49 rank-1 terms for Strassen 4x4*4x4.
    
    Returns list of (u_dict, v_dict, w_dict) where:
      u_dict: {(i,j): coeff} for A entries
      v_dict: {(j,k): coeff} for B entries  
      w_dict: {(i,k): coeff} for C entries
    """
    terms = []
    
    # Strassen 2x2 meta-terms: (A_local, B_local, C_local)
    # A_local: {(li,lj): coeff}, B_local: {(lj,lk): coeff}, C_local: {(li,lk): coeff}
    strassen_meta = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # 7 block-level Strassen multiplications
    # Each: (A_blocks, B_blocks, C_blocks)
    # A_blocks: [(br, bc, coeff)], B_blocks: [(br, bc, coeff)], C_blocks: {(br,bc): coeff}
    block_mults = [
        # M1 = (A11+A22)(B11+B22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0):1, (1,1):1}),
        # M2 = (A21+A22)B11
        ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
        # M3 = A11(B12-B22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1):1, (1,1):1}),
        # M4 = A22(B21-B11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0):1, (1,0):1}),
        # M5 = (A11+A12)B22
        ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
        # M6 = (A21-A11)(B11+B12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1):1}),
        # M7 = (A12-A22)(B21+B22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0):1}),
    ]
    
    for A_blocks, B_blocks, C_blocks in block_mults:
        for A_meta, B_meta, C_meta in strassen_meta:
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for (li, lj), meta_c in A_meta.items():
                for (br, bc, block_c) in A_blocks:
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + meta_c * block_c
            
            for (lj, lk), meta_c in B_meta.items():
                for (br, bc, block_c) in B_blocks:
                    gj = br * 2 + lj
                    gk = bc * 2 + lk
                    v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + meta_c * block_c
            
            for (li, lk), meta_c in C_meta.items():
                for (br, bc), block_c in C_blocks.items():
                    gi = br * 2 + li
                    gk = bc * 2 + lk
                    w_dict[(gi, gk)] = w_dict.get((gi, gk), 0) + meta_c * block_c
            
            terms.append((u_dict, v_dict, w_dict))
    
    return terms

# Precompute once
_STRASSEN_4X4_TERMS = _build_strassen_4x4_terms()

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add 49 Strassen terms for a 4x4*4x4 block."""
        for u_dict, v_dict, w_dict in _STRASSEN_4X4_TERMS:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            for (gi, gj), coeff in u_dict.items():
                u_vec[(A_base_i + gi) * m + (A_base_j + gj)] = coeff
            
            for (gj, gk), coeff in v_dict.items():
                v_vec[(B_base_j + gj) * p + (B_base_k + gk)] = coeff
            
            for (gi, gk), coeff in w_dict.items():
                w_vec[(C_base_k + gk) * n + (C_base_i + gi)] = coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B1 (columns 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (columns 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (column 8) - schoolbook, 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    # Build U, V, W matrices
    rank = len(U_cols)
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for col in range(rank):
        for row in range(16):
            U[row, col] = U_cols[col][row]
        for row in range(36):
            V[row, col] = V_cols[col][row]
            W[row, col] = W_cols[col][row]
    
    # Fast verification using tensor contraction
    if time.time() < deadline - 10:
        # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # Use einsum for efficient computation
        U_r = U.reshape(n, m, rank)  # (4, 4, rank)
        V_r = V.reshape(m, p, rank)  # (4, 9, rank)
        W_r = W.reshape(p, n, rank)  # (9, 4, rank)
        
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # This is a 3-index tensor. Compute via einsum:
        # First compute intermediate: M[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r]
        # Then T[i,j,k] = sum_k M[i,j,k] * W_r[k,i,r]... no that's not right.
        
        # Better: T[i,j,k] = sum_r (U_r[i,j,r] * W_r[k,i,r]) * V_r[j,k,r]
        # Let P[i,k,r] = U_r[i,j,r]... no, j is free.
        
        # Actually: T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # = sum_{j,r} ... no, j is a free index.
        
        # Let's compute it as: for each r, form outer product contributions.
        # Or use: T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # But that's O(n*m*p*rank) which is 4*4*9*114 = 16416 operations - fine.
        
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        
        valid = np.allclose(T, 1.0, atol=1e-10)
        
        if not valid:
            # Fallback to schoolbook
            U_cols.clear()
            V_cols.clear()
            W_cols.clear()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = [0] * 16
                        v_vec = [0] * 36
                        w_vec = [0] * 36
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k] = 1
                        w_vec[k * n + i] = 1
                        add_term(u_vec, v_vec, w_vec)
            
            rank = len(U_cols)
            U = np.zeros((16, rank), dtype=int)
            V = np.zeros((36, rank), dtype=int)
            W = np.zeros((36, rank), dtype=int)
            for col in range(rank):
                for row in range(16):
                    U[row, col] = U_cols[col][row]
                for row in range(36):
                    V[row, col] = V_cols[col][row]
                    W[row, col] = W_cols[col][row]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
