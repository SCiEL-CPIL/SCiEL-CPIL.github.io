import time
import numpy as np

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
    - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
    - Total rank: 49 + 49 + 16 = 114
    """
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))
    
    # Strassen 2x2: 7 terms, each (u_dict, v_dict, w_dict)
    # u: {local_A_idx: coeff}, v: {local_B_idx: coeff}, w: {local_C_idx: coeff}
    # Local A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),       # m1=(a+d)(e+h) -> c11+c22
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),            # m2=(c+d)e -> c21-c22
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),            # m3=a(f-h) -> c12+c22
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),            # m4=d(g-e) -> c11+c21
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),            # m5=(a+b)h -> -c11+c12
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),            # m6=(c-a)(e+f) -> c22
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),            # m7=(b-d)(g+h) -> c11
    ]
    
    def add_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """
        Add 49 terms for 4x4 * 4x4 -> 4x4 block multiplication.
        
        A occupies rows [A_i_base, A_i_base+3], cols [A_j_base, A_j_base+3]
        B occupies rows [B_j_base, B_j_base+3], cols [B_k_base, B_k_base+3]
        C occupies rows [C_i_base, C_i_base+3], cols [C_k_base, C_k_base+3]
        """
        # Top-level Strassen: 7 block multiplications
        # Each is (sum of A-blocks) * (sum of B-blocks) -> sum of C-blocks
        # A-blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1) in 2x2 grid of 2x2 blocks
        # B-blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
        # C-blocks: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
        
        # Top-level Strassen terms: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # Each coeff is +1 or -1
        top_strassen = [
            # M1 = (A11+A22)(B11+B22) -> C11 + C22
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            # M2 = (A21+A22)B11 -> C21 - C22
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            # M3 = A11(B12-B22) -> C12 + C22
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            # M4 = A22(B21-B11) -> C11 + C21
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            # M5 = (A11+A12)B22 -> -C11 + C12
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            # M6 = (A21-A11)(B11+B12) -> C22
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            # M7 = (A12-A22)(B21+B22) -> C11
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
        ]
        
        for a_block_coeffs, b_block_coeffs, c_block_coeffs in top_strassen:
            # For each inner Strassen term, build U, V, W vectors
            for u_inner, v_inner, w_inner in strassen_2x2:
                u_vec = [0] * nm
                v_vec = [0] * mp
                w_vec = [0] * pn
                
                # Fill U: for each A-block with nonzero coeff, for each local entry in u_inner
                for block_idx, block_coeff in a_block_coeffs.items():
                    block_row = block_idx // 2
                    block_col = block_idx % 2
                    for local_idx, inner_coeff in u_inner.items():
                        li = local_idx // 2
                        lj = local_idx % 2
                        gi = A_i_base + block_row * 2 + li
                        gj = A_j_base + block_col * 2 + lj
                        u_vec[gi * m + gj] += block_coeff * inner_coeff
                
                # Fill V: for each B-block with nonzero coeff, for each local entry in v_inner
                for block_idx, block_coeff in b_block_coeffs.items():
                    block_row = block_idx // 2
                    block_col = block_idx % 2
                    for local_idx, inner_coeff in v_inner.items():
                        lj = local_idx // 2
                        lk = local_idx % 2
                        gj = B_j_base + block_row * 2 + lj
                        gk = B_k_base + block_col * 2 + lk
                        v_vec[gj * p + gk] += block_coeff * inner_coeff
                
                # Fill W: for each C-block with nonzero coeff, for each local entry in w_inner
                for block_idx, block_coeff in c_block_coeffs.items():
                    block_row = block_idx // 2
                    block_col = block_idx % 2
                    for local_idx, inner_coeff in w_inner.items():
                        li = local_idx // 2
                        lk = local_idx % 2
                        gi = C_i_base + block_row * 2 + li
                        gk = C_k_base + block_col * 2 + lk
                        w_vec[gk * n + gi] += block_coeff * inner_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # A * B[:,0:4] -> C[:,0:4]
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # A * B[:,4:8] -> C[:,4:8]
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # A * B[:,8] -> C[:,8] - schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * nm
            v_vec = [0] * mp
            w_vec = [0] * pn
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    # Build output matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(nm)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(mp)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(pn)]
    
    return {"U": U, "V": V, "W": W}
