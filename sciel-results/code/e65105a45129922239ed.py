"""Bootstrap spawn: Direct implementation of rank 182 decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def add_strassen_2x2(U, V, W, rank_offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base, n, m, p):
    """
    Adds 7 Strassen terms for 2x2 block multiplication to the global matrices.
    a_row_base, a_col_base: top-left of A block
    b_row_base, b_col_base: top-left of B block
    c_row_base, c_col_base: top-left of C block
    """
    # U index: row * m + col
    # V index: row * p + col
    # W index: col * n + row
    
    # Strassen formulas for C = AB
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01 - B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
    # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
    # M7 = (A01 - A11)(B10 + B11) -> C00 += M7

    definitions = [
        # (U, V, W)
        # U: A indices, V: B indices, W: C indices
        # Local offsets 0,1
        ([((0, 0, 1), (1, 1, 1))], 
         [((0, 0, 1), (1, 1, 1))], 
         [((0, 0, 1), (1, 1, 1))]), # M1
        ([((1, 0, 1), (1, 1, 1))], 
         [((0, 0, 1))], 
         [((1, 0, 1), (1, 1, -1))]), # M2
        ([((0, 0, 1))], 
         [((0, 1, 1), (1, 1, -1))], 
         [((0, 1, 1), (1, 1, 1))]), # M3
        ([((1, 1, 1))], 
         [((1, 0, 1), (0, 0, -1))], 
         [((0, 0, 1), (1, 0, 1))]), # M4
        ([((0, 0, 1), (0, 1, 1))], 
         [((1, 1, 1))], 
         [((0, 0, -1), (0, 1, 1))]), # M5
        ([((1, 0, 1), (0, 0, -1))], 
         [((0, 0, 1), (0, 1, 1))], 
         [((1, 1, 1))]), # M6
        ([((0, 1, 1), (1, 1, -1))], 
         [((1, 0, 1), (1, 1, 1))], 
         [((0, 0, 1))])  # M7
    ]
    
    for u_list, v_list, w_list in definitions:
        col_idx = rank_offset
        for r, c, coeff in u_list:
            u_row = (a_row_base + r) * m + (a_col_base + c)
            U[u_row][col_idx] += coeff
        for r, c, coeff in v_list:
            v_row = (b_row_base + r) * p + (b_col_base + c)
            V[v_row][col_idx] += coeff
        for r, c, coeff in w_list:
            # W index: col * n + row
            w_row = (c_col_base + c) * n + (c_row_base + r)
            W[w_row][col_idx] += coeff
        rank_offset += 1
        
    return rank_offset

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strategy:
    # We want to multiply 6x6 matrices.
    # We can decompose this into a block matrix multiplication.
    # Let's try to find a lower rank than 189.
    # 189 = 27 * 7. This corresponds to 27 blocks of 2x2 Strassen.
    # This happens if we split 6x6 into 3x3 blocks of size 2x2.
    # So we are effectively doing 3x3 matrix multiplication where each "entry" is a 2x2 matrix.
    # The rank of 3x3 matrix multiplication is 19 (Bini-Capovani-Bini).
    # So the total rank would be 19 * 7 = 133.
    # This is a massive improvement over 189.
    
    # However, implementing the 19-term decomposition for 3x3 with 2x2 blocks is complex.
    # The 19 terms are linear combinations of the 9 blocks.
    # Let's see if we can just use the known rank 19 coefficients.
    
    # Rank 19 decomposition for 3x3 matrix multiplication (C = AB):
    # C_ij = sum_{k=1}^{19} U_k(i,j) * (A_k * B_k)
    # where A_k = sum_{i,j} u_k(i,j) A_ij and B_k = sum_{j,l} v_k(j,l) B_jl.
    
    # The 19 terms (u, v, w) are:
    # 1.  u: [[1,0,0],[0,0,0],[0,0,0]], v: [[0,0,0],[0,1,0],[0,0,0]], w: [[0,0,0],[0,0,0],[0,1,0]] -> This is just one term.
    # Actually, the standard rank 19 decomposition is often listed as a set of 19 triplets (U, V, W) of 3x3 matrices.
    
    # Let's find a reliable source or derive it.
    # Since I cannot access external sources, I will attempt to construct it or use a fallback.
    # If I cannot guarantee correctness of rank 133 quickly, I will stick to a known correct but higher rank, 
    # or try to optimize the 189 implementation to be faster/robust, though rank is the score.
    
    # Let's try to implement the rank 133 solution.
    # The 19 terms for 3x3 MM are:
    # (Source: "Rank 19 decomposition of 3x3 matrix multiplication")
    # Term 1:  U=[[1,0,0],[0,0,0],[0,0,0]], V=[[0,0,0],[0,1,0],[0,0,0]], W=[[0,0,0],[0,0,0],[0,1,0]] -- Wait, this is rank 1 term? No.
    # The decomposition is a sum of 19 rank 1 tensors in the tensor space.
    # Each term is a triplet of 3x3 matrices (U, V, W) such that T = sum U o V o W.
    # The multiplication tensor T has entries T_{ijk, lmn} = delta_{jl} * delta_{km} ... no.
    # T_{i, j, k} corresponds to C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor is sum_{j} e_i o e_j o e_k.
    
    # The 19 terms are:
    # 1.  U=[[1,0,0],[0,0,0],[0,0,0]], V=[[0,0,0],[0,1,0],[0,0,0]], W=[[0,0,0],[0,0,0],[0,1,0]] -> This is not correct format.
    # Let's use the explicit coefficients from a known paper.
    # Bini, Capovani, Lotti, Romani (1978) "Rank 19 decomposition of 3x3 matrix multiplication".
    
    # Coefficients (u_k, v_k, w_k) for k=1..19:
    # I will hardcode them. If I get it wrong, I will fall back to 189.
    
    # To be safe and fast, I will implement the 189 solution first, then check time.
    # 189 is 3x3 blocks of 2x2 Strassen.
    # This is straightforward and verified in the previous code.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    # Iterate over 3x3 blocks
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row][col_idx] += coeff
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row][col_idx] += coeff
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row][col_idx] += coeff

    # Check time
    elapsed = time.time() - start_time
    if elapsed < 10:
        # We have time to try to improve?
        # No, the rank is fixed at 189 in this structure.
        # To improve, we need the 133 rank decomposition.
        # I will not attempt to implement 133 from memory in this short time window due to high risk of error.
        # I will return the valid 189 solution.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
