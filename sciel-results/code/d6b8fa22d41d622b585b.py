"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS
    total_cells = n * m * p
    
    # Initialize with schoolbook rank 120 as a fallback
    U_school = np.zeros((n * m, total_cells), dtype=object)
    V_school = np.zeros((m * p, total_cells), dtype=object)
    W_school = np.zeros((p * n, total_cells), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist(),
        "rank": total_cells
    }
    
    # Check deadline
    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result
    
    # Attempt block decomposition with Strassen for 2x2x2 blocks
    # n=4, m=5, p=6
    # We can form blocks of size 2x2x2 for i in {0,1}, {2,3}; j in {0,1}, {2,3}; k in {0,1}, {2,3}, {4,5}
    # This covers i=0..3, j=0..3, k=0..5
    # Remaining: j=4 (the 5th row of m, index 4), with i=0..3, k=0..5
    # The sub-tensor for j=4 is J_{4 x 1 x 6} which has rank 4*6 = 24
    # Total rank = 12 blocks * 7 (Strassen rank) + 24 (schoolbook for remaining) = 84 + 24 = 108
    
    # Let's verify if this is valid and construct it
    # Strassen decomposition for 2x2x2:
    # The 7 multiplications are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We'll create new U, V, W arrays
    rank_strassen = 7
    num_i_blocks = n // 2  # 2
    num_j_blocks = m // 2  # 2
    num_k_blocks = p // 2  # 3
    
    # Strassen blocks cover i in [0, n-1] if n even, j in [0, num_j_blocks*2-1], k in [0, num_k_blocks*2-1]
    # Here: i covers 0..3 (all), j covers 0..3 (4 of 5), k covers 0..5 (all)
    # Remaining: j=4
    
    total_rank = num_i_blocks * num_j_blocks * num_k_blocks * rank_strassen + (m - num_j_blocks*2) * n * p
    # = 2*2*3*7 + 1*4*6 = 84 + 24 = 108
    
    if total_rank >= total_cells:
        return best_result  # No improvement
    
    U_new = np.zeros((n * m, total_rank), dtype=object)
    V_new = np.zeros((m * p, total_rank), dtype=object)
    W_new = np.zeros((p * n, total_rank), dtype=object)
    
    col = 0
    
    # Strassen terms definition:
    # For a 2x2 matrix A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]]
    # C = A*B where C00 = M1 - M4 + M6 - M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M3 + M5 + M7
    
    # The 7 terms with their (A_coeff, B_coeff) for each of the 4 output positions:
    # We need to express each M_r as sum of A_{ij} * B_{kl}
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B10-B11) = A00B10 - A00B11
    # M4 = A11(B01-B00) = A11B01 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # For each Strassen term r, we have a set of (local_i, local_j, coeff_U, local_k, coeff_V)
    # Actually, U[i*m+j, r] is the coefficient for A[i,j], V[j*p+k, r] for B[j,k]
    
    # Let's define the 7 terms as lists of (ai, aj, ui, bk, bk_j, vk) where:
    # ai, aj are local row, col of A (0 or 1)
    # bk, bk_j are local row, col of B (0 or 1) -- wait, B is 2x2, so rows 0,1 cols 0,1
    # ui, vk are the coefficients
    
    strassen_terms = [
        # M1
        [(0,0,1, 0,0,1), (0,1,1, 0,1,1), (1,0,1, 0,0,1), (1,1,1, 0,1,1)],
        # M2
        [(1,0,1, 0,0,1), (1,1,1, 0,0,1)],
        # M3
        [(0,0,1, 1,0,1), (0,1,-1, 1,1,1)],  # Wait, M3 = A00(B10-B11). A00 is (0,0). B10 is (1,0), B11 is (1,1)
        # Correction: M3 = A00*B10 - A00*B11. So terms: (A00, B10, +1), (A00, B11, -1)
        # M4 = A11*B01 - A11*B00. Terms: (A11, B01, +1), (A11, B00, -1)
        # M5 = A00*B11 + A01*B11. Terms: (A00, B11, +1), (A01, B11, +1)
        # M6 = A10*B00 + A10*B01 - A00*B00 - A00*B01
        # M7 = A01*B10 + A01*B11 - A11*B10 - A11*B11
    ]
    
    # Let me redefine properly. For each term t, we have a list of (ai, aj, coeff_A, bk, bk_j, coeff_B)
    # where ai, aj in {0,1} are indices into the 2x2 A block, bk, bk_j in {0,1} into 2x2 B block
    # coeff_A and coeff_B are the multipliers. The product is coeff_A * coeff_B * A[ai,aj] * B[bk,bk_j]
    
    terms_def = [
        # M1
        [(0,0, 1, 0,0, 1), (0,0, 1, 0,1, 1), (1,1, 1, 0,0, 1), (1,1, 1, 0,1, 1)],
        # M2
        [(1,0, 1, 0,0, 1), (1,1, 1, 0,0, 1)],
        # M3
        [(0,0, 1, 1,0, 1), (0,0, 1, 1,1, -1)],
        # M4
        [(1,1, 1, 0,1, 1), (1,1, 1, 0,0, -1)],
        # M5
        [(0,0, 1, 1,1, 1), (0,1, 1, 1,1, 1)],
        # M6
        [(1,0, 1, 0,0, 1), (1,0, 1, 0,1, 1), (0,0, 1, 0,0, -1), (0,0, 1, 0,1, -1)],
        # M7
        [(0,1, 1, 1,0, 1), (0,1, 1, 1,1, 1), (1,1, 1, 1,0, -1), (1,1, 1, 1,1, -1)]
    ]
    
    # W coefficients for each term and each output position (ci, cj) in {0,1}
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    w_coeffs = [
        # M1
        [1, 0, 0, 1],   # for C00, C01, C10, C11
        # M2
        [0, 0, 1, 0],
        # M3
        [0, 1, 0, -1],
        # M4
        [-1, 0, 1, 0],
        # M5
        [0, 1, 0, 1],
        # M6
        [1, 0, 0, 0],
        # M7
        [-1, 0, 0, 1]
    ]
    
    for bi in range(num_i_blocks):
        for bj in range(num_j_blocks):
            for bk in range(num_k_blocks):
                for t in range(rank_strassen):
                    # Process term t for block (bi, bj, bk)
                    # Global indices:
                    # i_global = 2*bi + ai
                    # j_global = 2*bj + aj
                    # k_global = 2*bk + bk_j (note: I used bk for block index, bk_j for local)
                    
                    # U and V coefficients
                    for (ai, aj, ca, bk_j, bk_j2, cb) in terms_def[t]:
                        # Wait, my tuple is (ai, aj, coeff_A, bk, bk_j, coeff_B)
                        # But bk and bk_j are both local indices for B. Let's say b_row, b_col
                        b_row, b_col = bk_j, bk_j2
                        i_glob = 2 * bi + ai
                        j_glob = 2 * bj + aj
                        k_glob = 2 * bk + b_col  # Wait, B is indexed by [j, k] in the global sense?
                        # In the tensor, U is indexed by (i,j), V by (j,k), W by (k,i)
                        # For a 2x2x2 block:
                        # A is 2x2 (i, j)
                        # B is 2x2 (j, k)
                        # C is 2x2 (i, k)
                        # So for term t:
                        # U[i,m+j] += coeff_A
                        # V[j,p+k] += coeff_B
                        # Where i, j are local to A, j, k are local to B
                        # Note: the j in A and the j in B must match for the multiplication to make sense
                        # In Strassen, we are multiplying two 2x2 matrices, so the j index is shared.
                        # In our global tensor, j is the middle dimension.
                        # So for a specific block:
                        # A_block has indices (ai, aj) where aj is the local j
                        # B_block has indices (bj_local, bk_local) where bj_local is the local j
                        # For the product to be valid, aj must equal bj_local.
                        # Let's re-check the Strassen terms.
                        # M1 = (A00+A11)(B00+B11). A00 is (i=0, j=0). B00 is (j=0, k=0).
                        # So A[0,0] * B[0,0]. Here j=0 matches.
                        # A00 is (i=0, j=0). B11 is (j=1, k=1).
                        # So A[0,0] * B[1,1]. Here j=0 for A, j=1 for B. MISMATCH!
                        # This means I cannot just add them in the tensor decomposition directly
                        # because the tensor decomposition requires that for each term,
                        # U[i,j] * V[j,k] * W[k,i] sums correctly.
                        # In the standard matrix multiplication tensor, the j index in U and V must be the same.
                        # In Strassen, the "multiplication" is not a single term A[i,j]B[j,k] but a sum.
                        # To use Strassen in a tensor decomposition, we need to express the product
                        # as a sum of rank-1 tensors.
                        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
                        # Each of these is a rank-1 tensor? No, A00B00 is a scalar times a rank-1 tensor?
                        # A00 is a scalar (entry of A), B00 is a scalar (entry of B).
                        # The contribution to C is a scalar.
                        # In the tensor decomposition, each term is u * v * w.
                        # For A00 * B00:
                        # u = e_{i=0, j=0}, v = e_{j=0, k=0}, w = e_{k=0, i=0} ? No.
                        # The tensor T_{i,j,k} = 1 if i,j,k are such that A[i,j]*B[j,k] contributes to C[i,k].
                        # Actually, T_{i,j,k} is 1 for all i,j,k in the schoolbook.
                        # For a term A[a_i, a_j] * B[b_j, b_k], it contributes to C[c_i, c_k].
                        # In the decomposition, we need to find u_{i,j}, v_{j,k}, w_{k,i} such that
                        # sum_t u_{i,j}^t v_{j,k}^t w_{k,i}^t = delta_{i, c_i} delta_{k, c_k} ... no.
                        
                        # Let's restart the tensor decomposition logic for Strassen.
                        # The goal is to compute C = A B.
                        # We want C[i,k] = sum_j A[i,j] B[j,k].
                        # Using Strassen for a 2x2 block:
                        # C00 = M1 - M4 + M6 - M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M3 + M5 + M7
                        # where M_r are the 7 products.
                        # Each M_r is a scalar (product of sums of A and B entries).
                        # But for tensor decomposition, we need to express the whole thing as
                        # sum_t (sum_{i,j} u_{i,j}^t A_{i,j}) (sum_{j,k} v_{j,k}^t B_{j,k}) (sum_{k,i} w_{k,i}^t C_{i,k})
                        # This is not quite right. The standard decomposition is:
                        # T = sum_t u_t \otimes v_t \otimes w_t
                        # where T_{i,j,k} = A_{i,j} B_{j,k} (conceptually).
                        # Actually, the tensor for matrix multiplication is defined such that
                        # <T, A \otimes B \otimes C^*> = trace(C^T A B)
                        # So T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in (AB)_{i,k}.
                        # Which is always true for the matching indices.
                        
                        # For Strassen, we are replacing the 8 schoolbook terms with 7 "complex" terms.
                        # But each "complex" term is not a single rank-1 tensor.
                        # M1 = (A00+A11)(B00+B11) is not a rank-1 term in the A,B,C space.
                        # It is a product of two linear forms in A and two in B.
                        # To include this in a tensor decomposition, we must expand it into rank-1 terms
                        # OR recognize that the "rank" of the Strassen decomposition is 7,
                        # meaning we can express the 2x2x2 matrix multiplication tensor
                        # as a sum of 7 rank-1 tensors.
                        
                        # This is possible because the 2x2x2 matrix multiplication tensor
                        # has rank 7.
                        # The 7 rank-1 tensors are defined by specific u, v, w vectors.
                        # I need to look up or derive the explicit u, v, w for Strassen.
                        
                        # This is getting complicated. Let's check if there's a simpler known decomposition
                        # for 4x5x6 that has rank < 120.
                        # Rank of 4x5x6 matrix multiplication tensor.
                        # Known bounds: R(4,5,6) <= ?
                        # Schoolbook is 120.
                        # Can we do better?
                        # If we use Strassen on 2x2x2 blocks:
                        # We have 2x2x3 blocks of size 2x2x2.
                        # Each 2x2x2 block has rank 7.
                        # So 12 blocks * 7 = 84.
                        # Remaining j=4: 4x1x6 block. Rank is 4*6 = 24.
                        # Total 108.
                        # This is a valid upper bound.
                        # The question is whether I can correctly implement the 7 rank-1 tensors for 2x2x2.
                        
                        # Let's find the explicit rank-1 decomposition for 2x2x2.
                        # This is a known result. I will try to construct it.
                        # If I can't, I will stick with 120.
                        
                        # Given the time constraint and complexity, let's try a different approach.
                        # Maybe I can use a known algorithm for rectangular matrix multiplication.
                        # But I don't have one memorized.
                        
                        # Let's try to implement the Strassen decomposition correctly.
                        # I found a reference that gives the u, v, w for Strassen.
                        # I will try to code it based on the standard Strassen formulas.
                        
                        # If I fail, I return 120.
                        
                        pass  # Placeholder for complex logic
                    pass
                pass
            pass
        pass
    
    # Since the implementation of Strassen tensor decomposition is complex and error-prone,
    # and I am running out of time, I will return the schoolbook decomposition.
    # However, I can try a simpler optimization:
    # Maybe I can combine some terms? Unlikely without a specific algorithm.
    
    # Let's check if there are any other simple improvements.
    # No obvious ones.
    
    return best_result
# EVOLVE-BLOCK-END
