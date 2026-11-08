import numpy as np
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U: (4, 7) - A side, indexed by [i*2+j]
    V: (4, 7) - B side, indexed by [j*2+k]  
    W: (4, 7) - C side, indexed by [k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[3,6] = 1
    
    return U, V, W

def _build_4x4x4():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses the tensor product structure: 4x4 = (2x2) ⊗ (2x2).
    
    U: (16, 49) - A side, indexed by [i*4+j]
    V: (16, 49) - B side, indexed by [j*4+k]
    W: (16, 49) - C side, indexed by [k*4+i]
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # For each outer Strassen term s (0..6) and inner Strassen term t (0..6):
    # The combined term index is s*7 + t
    # 
    # The 4x4 multiplication can be viewed as:
    # A (4x4) = A_outer (2x2 of 2x2 blocks)
    # B (4x4) = B_outer (2x2 of 2x2 blocks)
    # C (4x4) = C_outer (2x2 of 2x2 blocks)
    #
    # The outer Strassen gives 7 terms, each computing a product of 2x2 block matrices.
    # Each such product is itself a 2x2x2 multiplication (rank 7).
    #
    # For outer term s and inner term t:
    # - U contribution: For each (i_l, j_l) in the 2x2 A-block, the coefficient is
    #   U2[i_l*2+j_l, t] * (coefficient from outer Strassen for that block)
    # - Similarly for V and W
    #
    # The outer Strassen terms for 4x4 (treating 2x2 blocks as elements):
    # Let A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Let B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # For each outer term, we need to know which A-blocks and B-blocks are involved
    # and with what coefficients.
    
    # Define outer Strassen terms: (list of (block_idx, coeff) for A, list for B)
    # Block index: 0=A00, 1=A01, 2=A10, 3=A11 for A
    #              0=B00, 1=B01, 2=B10, 3=B11 for B
    
    outer_terms = [
        # s=0: (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # s=1: (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)]),
        # s=2: A00(B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)]),
        # s=3: A11(B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)]),
        # s=4: (A00+A01)B11
        ([(0, 1), (1, 1)], [(3, 1)]),
        # s=5: (A10-A00)(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # s=6: (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]
    
    for s in range(7):
        A_blocks, B_blocks = outer_terms[s]
        for t in range(7):
            col = s * 7 + t
            
            # For U: each A-block (block_idx, coeff) contributes
            # For the 2x2 block at position block_idx, the local indices (i_l, j_l)
            # map to global A index: (block_row*2 + i_l)*4 + (block_col*2 + j_l)
            # where block_row = block_idx // 2, block_col = block_idx % 2
            
            for (block_idx, coeff) in A_blocks:
                block_row = block_idx // 2
                block_col = block_idx % 2
                for i_l in range(2):
                    for j_l in range(2):
                        local_idx = i_l * 2 + j_l  # index into U2
                        if U2[local_idx, t] != 0:
                            global_row = (block_row * 2 + i_l) * 4 + (block_col * 2 + j_l)
                            U[global_row, col] += coeff * U2[local_idx, t]
            
            # For V: similar for B-blocks
            # B block (block_idx, coeff), local index (j_l, k_l) maps to
            # global B index: (block_row*2 + j_l)*4 + (block_col*2 + k_l)
            # V2 is indexed by [j*2+k]
            
            for (block_idx, coeff) in B_blocks:
                block_row = block_idx // 2
                block_col = block_idx % 2
                for j_l in range(2):
                    for k_l in range(2):
                        local_idx = j_l * 2 + k_l  # index into V2
                        if V2[local_idx, t] != 0:
                            global_row = (block_row * 2 + j_l) * 4 + (block_col * 2 + k_l)
                            V[global_row, col] += coeff * V2[local_idx, t]
            
            # For W: the output C-blocks. In Strassen, the C-blocks are computed as
            # linear combinations of the M terms. But in the tensor decomposition,
            # W just needs to match the C indices.
            # The C block for outer term s is determined by which C-blocks the M_s
            # contributes to. However, in the tensor decomposition framework,
            # each term contributes to specific C entries.
            # 
            # Actually, for the tensor decomposition, W[c_idx, col] gives the
            # coefficient for C[c_idx] in term col. The C indices are the same
            # structure as the 4x4 matrix.
            #
            # The key insight: for the 4x4x4 tensor, the W side is indexed by
            # [k*4+i] where k is the column of C and i is the row of C.
            # The outer Strassen determines which C-blocks are involved, but
            # actually each M_s is a full 2x2 matrix that gets placed into
            # specific C-block positions.
            #
            # Wait - I need to think about this more carefully. In the tensor
            # decomposition, each rank-1 term is u ⊗ v ⊗ w where:
            # - u is a vector in R^16 (A side)
            # - v is a vector in R^16 (B side)  
            # - w is a vector in R^16 (C side)
            # and the tensor is sum of u_r ⊗ v_r ⊗ w_r
            #
            # The tensor T[i,j,k] = A[i,j] * B[j,k] * C[k,i] (with appropriate
            # index conventions). For our case:
            # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
            # So T[i,j,k] = delta(i,j,k) in the sense that the tensor has
            # T[i,j,k] = 1 when all indices are valid (it's the multiplication tensor).
            #
            # Actually, the standard convention: the matrix multiplication tensor
            # T has T[i,j,k] = 1 for all valid i,j,k, and the decomposition
            # satisfies: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            #
            # So W is indexed by [k*n+i] = [k*4+i].
            #
            # For the outer Strassen term s, the C-blocks involved are determined
            # by the Strassen formula. But actually, in the tensor decomposition,
            # each term r contributes to all C entries via the product of the
            # U, V, W entries. The W vector for term r has non-zero entries at
            # specific [k*4+i] positions.
            #
            # For the 2x2 Strassen, W2 is indexed by [k*2+i] for the 2x2 result.
            # For the 4x4 case, we need to map this to the 4x4 C matrix.
            #
            # The C-block for outer term s: in Strassen, M_s contributes to
            # specific C-blocks. But in the tensor decomposition, we just need
            # W to have the right structure so that the sum works out.
            #
            # Let me think differently. The 4x4x4 tensor can be written as:
            # T = sum_{s=0}^{6} sum_{t=0}^{6} u_{s,t} ⊗ v_{s,t} ⊗ w_{s,t}
            # where u_{s,t} is built from the outer and inner Strassen A-coefficients,
            # v_{s,t} from B-coefficients, and w_{s,t} from C-coefficients.
            #
            # For w_{s,t}: the C-side coefficient. The outer Strassen term s
            # produces a 2x2 matrix M_s (in block space). This M_s is added to
            # C-blocks according to the Strassen reconstruction formula. But in
            # the tensor decomposition, we don't need to track the reconstruction;
            # we just need the tensor identity to hold.
            #
            # Actually, I realize the issue: the W vector for each term should be
            # such that when we sum U[i,j]*V[j,k]*W[k,i] over r, we get 1 for
            # all valid (i,j,k). 
            #
            # Let me verify with the 2x2 case first. For 2x2x2:
            # T[i,j,k] = 1 for all i,j,k in {0,1}
            # We need sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
            #
            # This is a standard result - the Strassen decomposition works.
            #
            # For 4x4x4, the tensor is T[i,j,k] = 1 for all i,j,k in {0,1,2,3}.
            # The decomposition should satisfy:
            # sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1
            #
            # Now, for the Kronecker product structure:
            # If we write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
            # Then the 4x4x4 tensor factors as:
            # T[i,j,k] = T2[i1,j1,k1] * T2[i0,j0,k0]
            # where T2 is the 2x2x2 tensor (all ones).
            #
            # So the 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors!
            # This means the rank-49 decomposition is the Kronecker product of
            # two rank-7 decompositions.
            #
            # For Kronecker product of tensors:
            # If T = T_a ⊗ T_b, and T_a = sum_r u_r ⊗ v_r ⊗ w_r (rank r_a)
            # and T_b = sum_s u'_s ⊗ v'_s ⊗ w'_s (rank r_b)
            # Then T = sum_{r,s} (u_r ⊗ u'_s) ⊗ (v_r ⊗ v'_s) ⊗ (w_r ⊗ w'_s)
            # with rank r_a * r_b.
            #
            # So for 4x4x4 = (2x2x2) ⊗ (2x2x2):
            # U[ (2*i1+i0)*4 + (2*j1+j0), s*7+t ] = U2[i1*2+j1, s] * U2[i0*2+j0, t]
            # V[ (2*j1+j0)*4 + (2*k1+k0), s*7+t ] = V2[j1*2+k1, s] * V2[j0*2+k0, t]
            # W[ (2*k1+k0)*4 + (2*i1+i0), s*7+t ] = W2[k1*2+i1, s] * W2[k0*2+i0, t]
            #
            # Let me implement this directly!
            
    # Redo with Kronecker product approach
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                for k0 in range(2):
                                    # Global indices
                                    i = 2*i1 + i0
                                    j = 2*j1 + j0
                                    k = 2*k1 + k0
                                    
                                    U_idx = i * 4 + j
                                    V_idx = j * 4 + k
                                    W_idx = k * 4 + i
                                    
                                    # Outer (block-level) Strassen indices
                                    U2_outer_idx = i1 * 2 + j1
                                    V2_outer_idx = j1 * 2 + k1
                                    W2_outer_idx = k1 * 2 + i1
                                    
                                    # Inner (within-block) Strassen indices
                                    U2_inner_idx = i0 * 2 + j0
                                    V2_inner_idx = j0 * 2 + k0
                                    W2_inner_idx = k0 * 2 + i0
                                    
                                    u_val = U2[U2_outer_idx, s] * U2[U2_inner_idx, t]
                                    v_val = V2[V2_outer_idx, s] * V2[V2_inner_idx, t]
                                    w_val = W2[W2_outer_idx, s] * W2[W2_inner_idx, t]
                                    
                                    U[U_idx, col] += u_val
                                    V[V_idx, col] += v_val
                                    W[W_idx, col] += w_val
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # Using einsum
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 4, 4, 8
    
    # Build 4x4x4 rank-49 decomposition
    U4, V4, W4 = _build_4x4x4()
    
    # Verify 4x4x4
    if not _verify(U4, V4, W4, 4, 4, 4):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # For 4x4x8: split B into two 4x4 halves
    # B_left: j in 0..3, k in 0..3 (columns 0-3)
    # B_right: j in 0..3, k in 4..7 (columns 4-7)
    # 
    # U is the same for both halves (A is 4x4)
    # V_left uses V4 with k in 0..3
    # V_right uses V4 with k in 4..7 (shifted)
    # W_left uses W4 for C columns 0..3
    # W_right uses W4 for C columns 4..7 (shifted)
    #
    # Total rank = 49 + 49 = 98
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))  # m*p = 4*8 = 32
    W = np.zeros((32, rank))  # p*n = 8*4 = 32
    
    # Left half (k = 0..3)
    U[:, 0:49] = U4
    # V for left half: V4 has shape (16, 49), indexed by j*4+k where k in 0..3
    # This is exactly what we need for V rows 0..15 (j*4+k for j in 0..3, k in 0..3)
    V[0:16, 0:49] = V4
    # W for left half: W4 has shape (16, 49), indexed by k*4+i where k in 0..3
    # This maps to W rows 0..15 (k*4+i for k in 0..3, i in 0..3)
    W[0:16, 0:49] = W4
    
    # Right half (k = 4..7)
    U[:, 49:98] = U4
    # V for right half: k in 4..7, so global V index = j*4 + (k) where k in 4..7
    # That is V rows 16..31: j*4 + (4 + k_local) where k_local in 0..3
    # V4[j*4+k_local, :] maps to V[j*4 + 4 + k_local, 49:98]
    V[16:32, 49:98] = V4
    # W for right half: k in 4..7, so global W index = k*4 + i where k in 4..7
    # That is W rows 16..31: (4+k_local)*4 + i = 16 + k_local*4 + i
    # W4[k_local*4+i, :] maps to W[16 + k_local*4 + i, 49:98]
    W[16:32, 49:98] = W4
    
    # Verify the full decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
