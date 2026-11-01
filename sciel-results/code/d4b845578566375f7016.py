import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    Implemented as a direct lookup table of the 49 non-zero tensor coordinates.
    
    The 4x4 multiplication tensor is decomposed using the 7x7 Kronecker product
    structure: 7 outer Strassen terms (2x2 block level) x 7 inner Strassen terms
    (scalar level within each 2x2 block) = 49 rank-1 terms.
    
    Index convention:
      U[i*4+k, r] : coefficient of A[i,k] in term r
      V[k*4+j, r] : coefficient of B[k,j] in term r  
      W[j*4+i, r] : coefficient for C[i,j] in term r (trace convention W[k*n+i,r])
    """
    start_time = time.time()
    
    RANK = 49
    
    # Precompute the 49 terms as explicit coordinate lists.
    # Each term r has: list of (U_idx, coeff), (V_idx, coeff), (W_idx, coeff)
    # U_idx = i*4+k for A[i,k]
    # V_idx = k*4+j for B[k,j]
    # W_idx = j*4+i for C[i,j]
    
    # Strassen 2x2 structure:
    # Outer (block-level) terms: which 2x2 blocks of A and B multiply
    # Inner (scalar-level) terms: which elements within those blocks multiply
    #
    # For outer term t, A blocks with coeff: a_blocks[t] = [(br,bc,coeff), ...]
    # For outer term t, B blocks with coeff: b_blocks[t] = [(br,bc,coeff), ...]
    # For outer term t, C blocks with coeff: c_blocks[t] = [(br,bc,coeff), ...]
    #
    # For inner term s, A elements with coeff: a_elems[s] = [(er,ec,coeff), ...]
    # For inner term s, B elements with coeff: b_elems[s] = [(er,ec,coeff), ...]
    # For inner term s, C elements with coeff: c_elems[s] = [(er,ec,coeff), ...]
    
    # Outer Strassen terms for 2x2 block multiplication:
    # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
    # P2 = (A21+A22)B11 -> C21 += P2, C22 -= P2
    # P3 = A11(B12-B22) -> C12 += P3, C22 += P3
    # P4 = A22(B21-B11) -> C11 += P4, C21 += P4
    # P5 = (A11+A12)B22 -> C11 -= P5, C12 += P5
    # P6 = (A21-A11)(B11+B12) -> C22 += P6
    # P7 = (A12-A22)(B21+B22) -> C11 += P7
    
    outer_a = [
        [(0,0,1),(1,1,1)],  # P1
        [(1,0,1),(1,1,1)],  # P2
        [(0,0,1)],          # P3
        [(1,1,1)],          # P4
        [(0,0,1),(0,1,1)],  # P5
        [(1,0,1),(0,0,-1)], # P6
        [(0,1,1),(1,1,-1)], # P7
    ]
    outer_b = [
        [(0,0,1),(1,1,1)],  # P1
        [(0,0,1)],          # P2
        [(0,1,1),(1,1,-1)], # P3
        [(1,0,1),(0,0,-1)], # P4
        [(1,1,1)],          # P5
        [(0,0,1),(0,1,1)],  # P6
        [(1,0,1),(1,1,1)],  # P7
    ]
    outer_c = [
        [(0,0,1),(1,1,1)],  # P1 -> C11, C22
        [(1,0,1),(1,1,-1)], # P2 -> C21, -C22
        [(0,1,1),(1,1,1)],  # P3 -> C12, C22
        [(0,0,1),(1,0,1)],  # P4 -> C11, C21
        [(0,0,-1),(0,1,1)], # P5 -> -C11, C12
        [(1,1,1)],          # P6 -> C22
        [(0,0,1)],          # P7 -> C11
    ]
    
    # Inner Strassen terms for 2x2 scalar multiplication:
    # P1 = (a+d)(e+h) -> C00 += P1, C11 += P1
    # P2 = (c+d)e -> C10 += P2, C11 -= P2
    # P3 = a(f-h) -> C01 += P3, C11 += P3
    # P4 = d(g-e) -> C00 += P4, C10 += P4
    # P5 = (a+b)h -> C00 -= P5, C01 += P5
    # P6 = (c-a)(e+f) -> C11 += P6
    # P7 = (b-d)(g+h) -> C00 += P7
    
    inner_a = [
        [(0,0,1),(1,1,1)],  # P1
        [(1,0,1),(1,1,1)],  # P2
        [(0,0,1)],          # P3
        [(1,1,1)],          # P4
        [(0,0,1),(0,1,1)],  # P5
        [(1,0,1),(0,0,-1)], # P6
        [(0,1,1),(1,1,-1)], # P7
    ]
    inner_b = [
        [(0,0,1),(1,1,1)],  # P1
        [(0,0,1)],          # P2
        [(0,1,1),(1,1,-1)], # P3
        [(1,0,1),(0,0,-1)], # P4
        [(1,1,1)],          # P5
        [(0,0,1),(0,1,1)],  # P6
        [(1,0,1),(1,1,1)],  # P7
    ]
    inner_c = [
        [(0,0,1),(1,1,1)],  # P1 -> C00, C11
        [(1,0,1),(1,1,-1)], # P2 -> C10, -C11
        [(0,1,1),(1,1,1)],  # P3 -> C01, C11
        [(0,0,1),(1,0,1)],  # P4 -> C00, C10
        [(0,0,-1),(0,1,1)], # P5 -> -C00, C01
        [(1,1,1)],          # P6 -> C11
        [(0,0,1)],          # P7 -> C00
    ]
    
    # Build U, V, W as dense 16x49 matrices
    U = np.zeros((16, RANK), dtype=float)
    V = np.zeros((16, RANK), dtype=float)
    W = np.zeros((16, RANK), dtype=float)
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # Accumulate U entries
            for (br, bc, bc_coeff) in outer_a[t]:
                for (er, ec, ec_coeff) in inner_a[s]:
                    row = 2 * br + er
                    col = 2 * bc + ec
                    U[row * 4 + col, r] += bc_coeff * ec_coeff
            
            # Accumulate V entries
            for (br, bc, bc_coeff) in outer_b[t]:
                for (er, ec, ec_coeff) in inner_b[s]:
                    row = 2 * br + er
                    col = 2 * bc + ec
                    V[row * 4 + col, r] += bc_coeff * ec_coeff
            
            # Accumulate W entries
            for (br, bc, bc_coeff) in outer_c[t]:
                for (er, ec, ec_coeff) in inner_c[s]:
                    row = 2 * br + er
                    col = 2 * bc + ec
                    W[col * 4 + row, r] += bc_coeff * ec_coeff
    
    # Verify with random test
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    # Vectorized verification:
    # For each term r: u_r = U[:,r]^T @ A_flat (length 16), v_r = V[:,r]^T @ B_flat
    # C[i,j] = sum_r W[j*4+i, r] * u_r * v_r
    A_flat = A_test.reshape(16)
    B_flat = B_test.reshape(16)
    
    # U @ A_flat gives vector of length RANK: (U @ A_flat)[r] = sum_{idx} U[idx,r] * A_flat[idx]
    u_vals = U.T @ A_flat  # shape (RANK,)
    v_vals = V.T @ B_flat  # shape (RANK,)
    
    # C[i,j] = sum_r W[j*4+i, r] * u_vals[r] * v_vals[r]
    # = sum_r W_flat[j*4+i, r] * (u_vals * v_vals)[r]
    # = (W * (u_vals * v_vals)[None, :]).sum(axis=1) reshaped
    wv = W * (u_vals * v_vals)[None, :]  # shape (16, RANK)
    C_flat = wv.sum(axis=1)  # shape (16,)
    C_computed = C_flat.reshape(4, 4)
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-9:
        # Fallback: trivial rank-64 decomposition
        rank_fallback = 64
        U_f = np.zeros((16, rank_fallback), dtype=float)
        V_f = np.zeros((16, rank_fallback), dtype=float)
        W_f = np.zeros((16, rank_fallback), dtype=float)
        
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U_f[i*4+k, r] = 1
                    V_f[k*4+j, r] = 1
                    W_f[j*4+i, r] = 1
        
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
