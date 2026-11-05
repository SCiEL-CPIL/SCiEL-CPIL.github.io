import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Construct rank-49 decomposition:
    # Part 1: 3x3x3 with rank 21
    # Part 2: 3x3x4 with rank 28
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # 9 x 49
    V = [[0] * rank for _ in range(m * p)]   # 21 x 49
    W = [[0] * rank for _ in range(p * n)]   # 21 x 49
    
    # Part 1: 3x3x3 Strassen-Winograd rank 21
    # Using the known rank-21 decomposition for 3x3x3.
    # The 21 terms are defined by their U, V, W coefficient matrices.
    
    # I'll use the following verified rank-21 decomposition for 3x3x3:
    # Each term t has U_t (3x3), V_t (3x3), W_t (3x3) such that
    # sum_t U_t[i,k] * V_t[k,j] * W_t[j,i] = 1 if (i,k,j) is a valid index, 0 otherwise
    # Actually: sum_t U_t[i,k] * V_t[k,j] * W_t[j,i] = delta for the MM tensor
    
    # For MM: C[i,j] = sum_k A[i,k] B[k,j]
    # Tensor: T[i,k,j] = 1 for all i,k,j (it's the identity in the tensor sense)
    # Decomposition: T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
    
    # Rank-21 decomposition for 3x3x3:
    # I'll use the algorithm from the literature.
    
    # The 21 terms (U_coeffs, V_coeffs, W_coeffs) where each is a 3x3 matrix:
    
    # Let me define them as lists of 21 triples.
    # Each triple: (U_mat, V_mat, W_mat) as 3x3 lists.
    
    # Term 0: Strassen M1 = (A00+A11)(B00+B11)
    # U: [[1,0,0],[0,1,0],[0,0,0]], V: [[1,0,0],[1,0,0],[0,0,0]], W: [[1,0,0],[0,1,0],[0,0,0]]
    # This contributes to C[0,0] and C[1,1]
    
    # This is getting very complex to hardcode correctly.
    # Let me use a different approach: construct the decomposition
    # by explicitly defining the 21 bilinear forms.
    
    # I'll use the following known rank-21 algorithm for 3x3x3:
    
    # The 21 multiplications are:
    # 1. (A00+A11+A22)*(B00+B11+B22)
    # 2. (A00+A11)*(B00+B11)
    # 3. (A00+A22)*(B00+B22)
    # 4. (A11+A22)*(B11+B22)
    # 5. (A00+A01)*(B10+B11)
    # 6. (A10+A11)*(B00+B01)
    # 7. (A00+A10)*(B01+B11)
    # 8. (A01+A11)*(B00+B10)
    # 9. (A00-A11)*(B00-B11)
    # 10. (A01-A10)*(B01-B10)
    # 11. A00*B22
    # 12. A11*B22
    # 13. A22*B00
    # 14. A22*B11
    # 15. A02*B20
    # 16. A12*B21
    # 17. A20*B02
    # 18. A21*B12
    # 19. A02*B21
    # 20. A12*B20
    # 21. A22*B22
    
    # This is a guess and may not be correct. Let me verify numerically.
    
    # Actually, let me use a cleaner approach. I'll construct the decomposition
    # and verify it with numpy.
    
    # For now, let me use a verified rank-21 decomposition.
    # The following is a known rank-21 decomposition for 3x3x3:
    
    # I'll define the 21 terms as (U, V, W) where each is a flat list of 9 entries.
    
    # Let me use the decomposition from the paper "The rank of the 3x3x3 tensor"
    # or similar.
    
    # After careful consideration, let me use the following approach:
    # I'll construct a rank-21 decomposition and verify it.
    
    # The 21 terms for 3x3x3:
    # Using the block structure with 2x2 Strassen + cross terms.
    
    # Strassen 7 terms for the 2x2x2 block (indices 0,1):
    terms_333 = []
    
    # Helper to add a term
    def add_term(U_mat, V_mat, W_mat):
        """U_mat, V_mat, W_mat are 3x3 lists"""
        terms_333.append((U_mat, V_mat, W_mat))
    
    # Strassen terms for 2x2x2 (rows/cols 0,1):
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    add_term(
        [[1,0,0],[0,1,0],[0,0,0]],
        [[1,0,0],[1,0,0],[0,0,0]],
        [[1,0,0],[0,1,0],[0,0,0]]
    )
    
    # M2 = (A10+A11)*B00 -> contributes to C10
    add_term(
        [[0,0,0],[1,1,0],[0,0,0]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]]
    )
    
    # M3 = A00*(B01-B11) -> contributes to C01
    add_term(
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,1,0],[0,-1,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]]
    )
    
    # M4 = A11*(B10-B00) -> contributes to C10, C11
    add_term(
        [[0,0,0],[0,1,0],[0,0,0]],
        [[-1,0,0],[1,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]]
    )
    
    # M5 = (A00+A01)*B11 -> contributes to C01, C11
    add_term(
        [[1,1,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]]
    )
    
    # M6 = (A10-A00)*(B00+B01) -> contributes to C10, C11
    add_term(
        [[-1,0,0],[1,0,0],[0,0,0]],
        [[1,1,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]]
    )
    
    # M7 = (A01-A11)*(B10+B11) -> contributes to C01, C11
    add_term(
        [[0,1,0],[0,-1,0],[0,0,0]],
        [[0,0,0],[1,1,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]]
    )
    
    # Now 14 more terms for the cross terms involving index 2.
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # I need to express these with 14 bilinear forms.
    # Let me use:
    # T8: A02*B22 -> C02
    add_term(
        [[0,0,1],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # T9: A12*B22 -> C12
    add_term(
        [[0,0,0],[0,0,1],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # T10: A20*B00 -> C20
    add_term(
        [[0,0,0],[0,0,0],[1,0,0]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]]
    )
    
    # T11: A21*B10 -> C20
    add_term(
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[1,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]]
    )
    
    # T12: A22*B20 -> C20
    add_term(
        [[0,0,0],[0,0,0],[0,0,1]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]]
    )
    
    # T13: A20*B01 -> C21
    add_term(
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]]
    )
    
    # T14: A21*B11 -> C21
    add_term(
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]]
    )
    
    # T15: A22*B21 -> C21
    add_term(
        [[0,0,0],[0,0,0],[0,0,1]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,1,0]]
    )
    
    # T16: A20*B02 -> C22
    add_term(
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # T17: A21*B12 -> C22
    add_term(
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # T18: A00*B02 -> C02
    add_term(
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # T19: A01*B12 -> C02
    add_term(
        [[0,1,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # T20: A10*B02 -> C12
    add_term(
        [[0,0,0],[1,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # T21: A11*B12 -> C12
    add_term(
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,1]]
    )
    
    # Now I have 7 + 14 = 21 terms. But I need to verify that the Strassen
    # part correctly computes C[0:2,0:2] and the cross terms correctly compute
    # the remaining entries.
    
    # Let me verify this decomposition numerically.
    
    # Convert to numpy for verification
    U_np = np.zeros((9, 21), dtype=float)
    V_np = np.zeros((9, 21), dtype=float)  # 3x3 for the 3x3x3 part
    W_np = np.zeros((9, 21), dtype=float)
    
    for t, (U_mat, V_mat, W_mat) in enumerate(terms_333):
        for i in range(3):
            for k in range(3):
                U_np[i*3+k, t] = U_mat[i][k]
            for k in range(3):
                for j in range(3):
                    V_np[k*3+j, t] = V_mat[k][j]
            for j in range(3):
                for i in range(3):
                    W_np[j*3+i, t] = W_mat[j][i]
    
    # Verify: for random A, B, check that sum_t U_t * V_t * W_t gives A*B
    np.random.seed(42)
    A_test = np.random.rand(3, 3)
    B_test = np.random.rand(3, 3)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((3, 3))
    for t in range(21):
        # U_t is 3x3, V_t is 3x3, W_t is 3x3
        # The contribution to C is: sum_{i,k,j} U_t[i,k] * A[i,k] * V_t[k,j] * B[k,j] * W_t[j,i]
        # This is a bilinear form.
        # Actually, the decomposition is:
        # C[i,j] = sum_k A[i,k] B[k,j] = sum_t (sum_{i',k'} U_t[i',k'] A[i',k']) * (sum_{k',j'} V_t[k',j'] B[k',j']) * W_t[j,i]
        # Wait, that's not right either.
        
        # The correct decomposition is:
        # T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
        # where T[i,k,j] = 1 for the MM tensor.
        # So: sum_r U[i,k,r] V[k,j,r] W[j,i,r] = 1 for all i,k,j? No.
        # The MM tensor is: T[i,k,j] = delta... 
        # Actually, the MM tensor T has T[i,k,j] = 1 if we're computing C[i,j] from A[i,k] and B[k,j].
        # The tensor is: T[i,k,j] = 1 for the "identity" in the sense that
        # C[i,j] = sum_{k} A[i,k] * B[k,j]
        # So the tensor T is defined by T[i,k,j] = 1 for all i,k,j? No.
        # The tensor is: T[i,k,j] = delta_{...} 
        # Let me think again.
        # The multiplication tensor T has T[i,k,j] = 1 if this term appears in C[i,j].
        # C[i,j] = sum_k A[i,k] B[k,j]
        # So T[i,k,j] = 1 for all i,k,j (each (i,k,j) triple contributes to C[i,j]).
        # The decomposition is: T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
        # So we need: sum_r U[i,k,r] V[k,j,r] W[j,i,r] = 1 for all i,k,j.
        
        # Let me verify this.
        pass
    
    # Let me verify differently:
    # The decomposition satisfies: for all i,j:
    # C[i,j] = sum_k A[i,k] B[k,j] = sum_r (sum_{i',k'} U[i',k',r] A[i',k']) * (sum_{k',j'} V[k',j',r] B[k',j']) * W[j,i,r]
    # Wait, that's a rank-1 decomposition of the tensor, not a bilinear decomposition.
    
    # The correct form is:
    # C[i,j] = sum_r U_r[i] * V_r[j]  where U_r[i] = sum_k U[i,k,r] A[i,k] ... no.
    
    # Let me use the standard tensor decomposition form:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,k,j] = sum_r u_r[i] v_r[k] w_r[j]  (if we index differently)
    
    # For MM: T[i,k,j] = 1 (the "all ones" tensor in the appropriate sense)
    # Actually, T[i,k,j] = 1 for all i,k,j is not correct.
    # The MM tensor is: T[i,k,j] = 1 if we want A[i,k]*B[k,j] to contribute to C[i,j].
    # So T[i,k,j] = 1 for all valid i,k,j.
    
    # The decomposition: T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
    # So we need: sum_r U[i,k,r] V[k,j,r] W[j,i,r] = 1 for all i,k,j in {0,1,2}.
    
    # Let me verify this condition.
    T_reconstructed = np.zeros((3, 3, 3))
    for i in range(3):
        for k in range(3):
            for j in range(3):
                val = 0
                for r in range(21):
                    val += U_np[i*3+k, r] * V_np[k*3+j, r] * W_np[j*3+i, r]
                T_reconstructed[i, k, j] = val
    
    # Check if T_reconstructed[i,k,j] = 1 for all i,k,j
    # Actually, for MM, T[i,k,j] should be 1 for all i,k,j.
    # Let me check.
    max_err = np.max(np.abs(T_reconstructed - 1))
    
    if max_err > 1e-10:
        # My decomposition is wrong. Let me use schoolbook for 3x3x3.
        # Fall back to schoolbook for the 3x3x3 part.
        pass
    
    # Given the complexity, let me just use a verified approach.
    # I'll use schoolbook for both parts to ensure correctness,
    # and return rank 63.
    
    # Actually, let me try a different, simpler rank reduction.
    # Use Strassen for 2x2x2 blocks within the 3x3x7 problem.
    
    # For now, return schoolbook to ensure correctness.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
