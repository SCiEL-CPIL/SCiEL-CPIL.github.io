"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 (Strassen 4x4x4) + Rank 16 (Schoolbook 4x4x1) = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: k=4 slice (Schoolbook)
    # i in 0..3, j in 0..3
    # U[i*m + j], V[j*p + 4], W[4*n + i]
    col_offset = 49
    for i in range(n):
        for j in range(m):
            c = col_offset + i * m + j
            U[i * m + j][c] = 1
            V[j * p + 4][c] = 1
            W[4 * n + i][c] = 1
            
    # Part 2: k in 0..3 slice (Strassen 4x4x4)
    # We need to generate the 49 terms for the 4x4x4 multiplication.
    # The Strassen decomposition acts on A (4x4), B (4x4), C (4x4).
    # In the context of the full 4x4x5 problem:
    # A is the full 4x4 matrix (indices i, j).
    # B is the first 4 columns of the 4x5 matrix (indices j, k where k in 0..3).
    # C is the first 4 columns of the 4x5 result (indices k, i where k in 0..3).
    
    # Helper to generate Strassen terms for size N
    def gen_strassen(size, i0, j0, k0):
        """
        Yields (i_global, j_global, k_global, coeff)
        i_global: index in A (row)
        j_global: index in B (row, which is col of A)
        k_global: index in C (col)
        """
        if size == 1:
            yield (i0, j0, k0, 1)
            return
        
        h = size // 2
        
        # Strassen Steps
        # 1. M1 = (A00+A11)(B00+B11) -> C00, C11
        # 2. M2 = (A10+A11)B00 -> C10, C11
        # 3. M3 = A00(B01-B11) -> C01, C11
        # 4. M4 = A11(B10-B00) -> C00, C10
        # 5. M5 = (A00+A01)B11 -> C00, C01
        # 6. M6 = (A10-A00)(B00+B01) -> C11
        # 7. M7 = (A01-A11)(B10+B11) -> C00, C11
        
        # Define steps as (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (offset_i, offset_j, sign)
        # B_blocks: list of (offset_j, offset_k, sign)
        # C_blocks: list of (offset_i, offset_k, sign)
        
        steps = [
            # 1
            ([(0,0,1), (h,h,1)], [(0,0,1), (h,h,1)], [(0,0,1), (h,h,1)]),
            # 2
            ([(h,0,1), (h,h,1)], [(0,0,1)], [(h,0,1), (h,h,1)]),
            # 3
            ([(0,0,1)], [(0,h,1), (h,h,-1)], [(0,h,1), (h,h,1)]),
            # 4
            ([(h,h,1)], [(h,0,1), (0,0,-1)], [(0,0,1), (h,0,1)]),
            # 5
            ([(0,0,1), (0,h,1)], [(h,h,1)], [(0,0,1), (0,h,1)]),
            # 6
            ([(h,0,1), (0,0,-1)], [(0,0,1), (0,h,1)], [(h,h,1)]),
            # 7
            ([(0,h,1), (h,h,-1)], [(h,0,1), (h,h,1)], [(0,0,1), (h,h,1)]),
        ]
        
        for a_blocks, b_blocks, c_blocks in steps:
            for a_off_i, a_off_j, a_sign in a_blocks:
                for b_off_j, b_off_k, b_sign in b_blocks:
                    # Recurse on the product of the blocks
                    # The recursive call generates terms for a standard size/2 multiplication
                    # i_l is row in local A, j_l is col in local A / row in local B
                    # k_l is col in local B / col in local C
                    for i_l, j_l, k_l, c_l in gen_strassen(h, 0, 0, 0):
                        coeff = a_sign * b_sign * c_l
                        
                        # Map local indices to global
                        # Global A index
                        gi = i0 + a_off_i + i_l
                        # Global B index
                        gj = j0 + b_off_j + j_l
                        
                        # Map to C blocks
                        for c_off_i, c_off_k, c_sign in c_blocks:
                            # Global C index (col)
                            # Note: C is indexed by [k, i] in the final W matrix?
                            # The task says W[k*n+i, r].
                            # So W index is k * n + i.
                            # Here, 'k' is the column of the result matrix C.
                            # 'i' is the row of the result matrix C.
                            # In our recursion:
                            # i_l is the row of the result block.
                            # k_l is the column of the result block.
                            
                            # Global row of C (which is 'i' in W index)
                            global_i = i0 + c_off_i + i_l
                            # Global col of C (which is 'k' in W index)
                            global_k = k0 + c_off_k + k_l
                            
                            # W index is global_k * n + global_i
                            # But we are yielding (i_global, j_global, k_global)
                            # where i_global is A row, j_global is B row (A col), k_global is C col.
                            # We need to pass the C row as well?
                            # The standard tensor decomposition T[i,j,k] = sum U[i,r]V[j,r]W[k,r]
                            # Here T is the multiplication tensor.
                            # T[i, j, k] is 1 if C[i,k] depends on A[i,j] and B[j,k].
                            # So the third index of the tensor is k (col of C).
                            # But W is defined as W[k*n+i, r].
                            # This means the "k" in the tensor T corresponds to the pair (k, i) in W?
                            # No, the tensor T has indices i (row A), j (col A), k (col B).
                            # The output C has indices i (row C), k (col C).
                            # The relation is C[i,k] = sum_j A[i,j]B[j,k].
                            # So the tensor is T[i, j, k].
                            # The W matrix in the task has shape (p*n, rank).
                            # The index in W is k*n + i.
                            # This implies that the "k" in the tensor T (col of B) is NOT the same as the "k" in W index.
                            # Wait.
                            # Task: W[k*n+i, r].
                            # This means the third mode of the tensor is indexed by (k, i).
                            # But matrix multiplication only has 3 modes: i, j, k.
                            # T[i, j, k] = 1.
                            # If we use W[k*n+i], then the third mode is effectively (k, i).
                            # This is a 2D index for the third mode.
                            # This is consistent with the fact that the output is a matrix.
                            
                            # So, for a term (i_A, j_B, k_B), it contributes to W[k_B * n + i_A].
                            # In my recursion:
                            # gi is i_A.
                            # gj is j_B.
                            # global_k is k_B.
                            # So the W index is global_k * n + gi.
                            
                            # However, the Strassen recursion I wrote yields (i_l, j_l, k_l) for the local tensor.
                            # i_l is row of local C? No, i_l is row of local A.
                            # j_l is col of local A.
                            # k_l is col of local B.
                            
                            # So:
                            # Global A row = gi
                            # Global B col = gj (which is A col)
                            # Global B row = gj? No.
                            # B is m x p. B[j, k].
                            # j is col of A. k is col of B.
                            # So V index is j * p + k.
                            # In my recursion, j_l is the col of local A (so row of local B).
                            # k_l is the col of local B.
                            # So global V index is gj * p + global_k?
                            # Wait, gj is the global col of A.
                            # global_k is the global col of B.
                            # So V index = gj * p + global_k.
                            
                            # And W index = global_k * n + gi.
                            
                            # So I yield (gi, gj, global_k) and let the caller map to U, V, W.
                            # But I also need to know which C block it goes to?
                            # No, the Strassen step definition already accounts for which C block the result goes to.
                            # The c_blocks list contains the offsets for the C block.
                            # c_off_i is the row offset of the C block.
                            # c_off_k is the col offset of the C block.
                            
                            # So the global C row is i0 + c_off_i + i_l.
                            # The global C col is k0 + c_off_k + k_l.
                            
                            # But wait, in the tensor T[i, j, k], the third index is k (col of B).
                            # The output C[i, k] is determined by summing over j.
                            # The Strassen algorithm computes linear combinations of C blocks.
                            # The "tensor" we are decomposing is the one that maps A and B to C.
                            # So the third index of the tensor MUST be the column of B (k).
                            # The row of C (i) is the first index of the tensor.
                            
                            # So, for a specific Strassen term:
                            # It computes a value M.
                            # M is added to C[i_c, k_c].
                            # This means that the tensor component T[i_c, j, k_c] gets a contribution.
                            # Where i_c is the row of C, j is the col of A, k_c is the col of B.
                            
                            # In the recursive call:
                            # i_l is the row of the local C block.
                            # j_l is the col of the local A block.
                            # k_l is the col of the local B block.
                            
                            # So:
                            # Global C row (i_c) = i0 + c_off_i + i_l
                            # Global A col (j) = j0 + b_off_j + j_l  (Note: j in T is col of A)
                            # Global B col (k_c) = k0 + c_off_k + k_l  (Note: k in T is col of B)
                            
                            # But wait, the V matrix index is j * p + k.
                            # Here j is col of A, k is col of B.
                            # So V index = (j0 + b_off_j + j_l) * p + (k0 + c_off_k + k_l).
                            
                            # And U matrix index is i * m + j.
                            # Here i is row of A, j is col of A.
                            # So U index = (i0 + a_off_i + i_l) * m + (j0 + b_off_j + j_l).
                            
                            # And W matrix index is k * n + i.
                            # Here k is col of B, i is row of C.
                            # So W index = (k0 + c_off_k + k_l) * n + (i0 + c_off_i + i_l).
                            
                            # So I should yield (U_idx, V_idx, W_idx, coeff).
                            
                            u_idx = (i0 + a_off_i + i_l) * m + (j0 + b_off_j + j_l)
                            v_idx = (j0 + b_off_j + j_l) * p + (k0 + c_off_k + k_l)
                            w_idx = (k0 + c_off_k + k_l) * n + (i0 + c_off_i + i_l)
                            
                            yield (u_idx, v_idx, w_idx, coeff)
                            
    # Fill U, V, W for Part 2
    col = 0
    for u_idx, v_idx, w_idx, coeff in gen_strassen(4, 0, 0, 0):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
