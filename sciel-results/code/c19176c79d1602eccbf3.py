"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm (rank 7).
    U: (4, 7) -> A indices (i,j)
    V: (4, 7) -> B indices (j,k)
    W: (4, 7) -> C indices (k,i) [Note: Task uses W[k*n+i, r]]
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i: 0*2+0=0, 1*2+0=2, 0*2+1=1, 1*2+1=3? 
    # Wait, W index is k*n + i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=0, i=1 -> 1
    # C[1,0] -> k=1, i=0 -> 2
    # C[1,1] -> k=1, i=1 -> 3
    # So W index mapping:
    # 0: C[0,0]
    # 1: C[0,1]
    # 2: C[1,0]
    # 3: C[1,1]
    
    # Strassen products M1..M7
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 -= M1 (in some conventions, let's use standard addition/subtraction)
    # Standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5
    
    # Let's assign columns 0..6 to M1..M7
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[3, 2] = -1 # C11
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1 # C00
    W[2, 3] = 1  # C10
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1 # C01
    W[3, 4] = 1 # C11
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1 # C00
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1 # C01
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We split the problem into:
    # 1. Multiplication for k in 0..3 (4x4x4 part)
    # 2. Multiplication for k in 4..6 (4x4x3 part)
    
    # Part 1: 4x4x4 using Strassen-Strassen (Rank 49)
    # We view 4x4 as 2x2 blocks of 2x2 matrices.
    # Outer Strassen: 7 products of 2x2 matrices.
    # Inner Strassen: Each 2x2 product uses rank 7.
    # Total rank 49.
    
    U_outer, V_outer, W_outer = _strassen_2x2_factors()
    
    # The "scalars" in the outer Strassen are actually 2x2 matrices.
    # So for each outer term r_out in 0..6, we have an inner decomposition.
    
    rank_part1 = 49
    U1 = np.zeros((n*m, rank_part1))
    V1 = np.zeros((m*4, rank_part1)) # k is 0..3 here
    W1 = np.zeros((4*n, rank_part1)) # k is 0..3 here
    
    # Mapping of indices for 4x4:
    # A[i,j] where i,j in 0..3.
    # Block I, J (I,J in 0..1).
    # A_block[I,J] contains A[2*I + i, 2*J + j] for i,j in 0..1.
    
    # For each outer term r_out, we have coefficients for the blocks.
    # U_outer[i_block, r_out] gives coeff for A_block.
    # V_outer[j_block, r_out] gives coeff for B_block.
    # W_outer[k_block, r_out] gives coeff for C_block.
    
    # We need to expand these block coefficients into element coefficients.
    # For a given outer term r_out, we use the inner Strassen decomposition.
    # The inner decomposition computes the product of two 2x2 matrices.
    # Let A_inner be the matrix formed by the linear combination of A_blocks specified by U_outer.
    # Let B_inner be the matrix formed by the linear combination of B_blocks specified by V_outer.
    # The product is C_inner.
    # Then C is updated by adding W_outer[k_block, r_out] * C_inner.
    
    # Actually, it's easier to think of the tensor as a Kronecker product of the two decompositions.
    # The rank-7 Strassen tensor for 2x2 is T.
    # The 4x4 tensor is T \otimes T.
    # The factors U, V, W are constructed by Kronecker products of the factors of T.
    
    U_base, V_base, W_base = _strassen_2x2_factors()
    
    # U_total for 4x4 part
    # U is (16, 49).
    # Index u_idx = (i_block * 2 + i_inner) * 4 + (j_block * 2 + j_inner) ?
    # No, U index is i*m + j = i*4 + j.
    # i = i_block * 2 + i_inner
    # j = j_block * 2 + j_inner
    # So u_idx = (i_block * 2 + i_inner) * 4 + (j_block * 2 + j_inner)
    
    # V index is j*p + k. Here p=4 for this part.
    # j = j_block * 2 + j_inner
    # k = k_block * 2 + k_inner
    # v_idx = (j_block * 2 + j_inner) * 4 + (k_block * 2 + k_inner)
    
    # W index is k*n + i. Here n=4.
    # k = k_block * 2 + k_inner
    # i = i_block * 2 + i_inner
    # w_idx = (k_block * 2 + k_inner) * 4 + (i_block * 2 + i_inner)
    
    # The rank index r is (r_outer * 7 + r_inner).
    
    for r_out in range(7):
        for r_in in range(7):
            r = r_out * 7 + r_in
            
            # Get coefficients from base factors
            # U_base[i_block, r_out]
            # V_base[j_block, r_out]
            # W_base[k_block, r_out]
            
            # And inner factors
            # U_base[i_inner, r_in]
            # V_base[j_inner, r_in]
            # W_base[k_inner, r_in]
            
            # The term in the decomposition is:
            # sum_{ib, jb, kb, ii, ji, ki} 
            #   U_base[ib, r_out] * U_base[ii, r_in] * A[ib*2+ii, jb*2+ji] *
            #   V_base[jb, r_out] * V_base[ji, r_in] * B[jb*2+ji, kb*2+ki] *
            #   W_base[kb, r_out] * W_base[ki, r_in] * C[kb*2+ki, ib*2+ii]
            
            # This can be written as a single rank-1 term in the 4x4 tensor if we combine the indices?
            # No, it's a sum of products. But since the factors are separable (Kronecker product structure),
            # we can just multiply the coefficients for the combined indices.
            
            for ib in range(2):
                for jb in range(2):
                    for kb in range(2):
                        for ii in range(2):
                            for ji in range(2):
                                for ki in range(2):
                                    coeff_u = U_base[ib, r_out] * U_base[ii, r_in]
                                    coeff_v = V_base[jb, r_out] * V_base[ji, r_in]
                                    coeff_w = W_base[kb, r_out] * W_base[ki, r_in]
                                    
                                    if coeff_u == 0 or coeff_v == 0 or coeff_w == 0:
                                        continue
                                    
                                    u_idx = (ib * 2 + ii) * 4 + (jb * 2 + ji)
                                    v_idx = (jb * 2 + ji) * 4 + (kb * 2 + ki)
                                    w_idx = (kb * 2 + ki) * 4 + (ib * 2 + ii)
                                    
                                    U1[u_idx, r] += coeff_u
                                    V1[v_idx, r] += coeff_v
                                    W1[w_idx, r] += coeff_w
                                    
    # Part 2: 4x4x3 using schoolbook (Rank 48)
    rank_part2 = 48
    U2 = np.zeros((n*m, rank_part2))
    V2 = np.zeros((m*3, rank_part2)) # k is 0..2 here (relative to part 2)
    W2 = np.zeros((3*n, rank_part2))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2[i * 4 + j, col] = 1
                V2[j * 3 + k, col] = 1
                W2[k * 4 + i, col] = 1
                col += 1
                
    # Combine parts
    total_rank = rank_part1 + rank_part2
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Copy Part 1
    U[:, :rank_part1] = U1
    V[:, :rank_part1] = V1 # V1 is (16, 49), V is (28, 49+48). V1 corresponds to k=0..3.
    # V indices for part 1: j*4 + k where k in 0..3.
    # V2 indices: j*3 + k where k in 0..2.
    # We need to map V2 to global V indices.
    # Global V index: j*7 + k_global.
    # For part 1: k_global = k_local (0..3).
    # For part 2: k_global = 4 + k_local (4..6).
    
    # So V1 is already correct for the first 4 columns of B?
    # V1 shape is (16, 49). Indices are j*4 + k.
    # But global V index is j*7 + k.
    # So we need to scatter V1 into V.
    
    # Let's re-construct V and W carefully.
    
    # V is (m*p, rank) = (28, 97).
    # V[j*7 + k, r]
    
    # For Part 1 (k=0..3):
    # V1[j*4 + k, r]
    # We need to place this at V[j*7 + k, r].
    for j in range(4):
        for k in range(4):
            V[j*7 + k, :rank_part1] = V1[j*4 + k, :rank_part1]
            
    # For Part 2 (k=4..6):
    # V2[j*3 + k_local, r]
    # k_global = 4 + k_local
    # Place at V[j*7 + (4 + k_local), r + rank_part1]
    for j in range(4):
        for k_local in range(3):
            V[j*7 + (4 + k_local), rank_part1:] = V2[j*3 + k_local, :]
            
    # W is (p*n, rank) = (28, 97).
    # W[k*4 + i, r]
    
    # For Part 1 (k=0..3):
    # W1[k*4 + i, r]
    # Place at W[k*4 + i, r].
    W[:16, :rank_part1] = W1
    
    # For Part 2 (k=4..6):
    # W2[k_local*4 + i, r]
    # k_global = 4 + k_local
    # Place at W[(4 + k_local)*4 + i, r + rank_part1]
    for k_local in range(3):
        W[(4 + k_local)*4 : (4 + k_local)*4 + 4, rank_part1:] = W2[k_local*4 : (k_local+1)*4, :]
        
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
