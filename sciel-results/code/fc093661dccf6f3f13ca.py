import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p  # 2*4*5 = 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = 130  # leave 20s margin

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Try to construct a rank 32 decomposition using Strassen
    # Structure:
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x5) = [B1, B2, B3] where B1 (4x2), B2 (4x2), B3 (4x1)
    # B1 = [B11; B21] where B11, B21 are 2x2
    # B2 = [B12; B22] where B12, B22 are 2x2
    # B3 = [B13; B23] where B13, B23 are 2x1
    
    # C (2x5)
    # C[:, 0:2] = A1 B11 + A2 B21
    # C[:, 2:4] = A1 B12 + A2 B22
    # C[:, 4:5] = A1 B13 + A2 B23
    
    # We use Strassen for the 2x2x2x2 blocks.
    # Strassen rank 7.
    # We have 4 such blocks: A1B11, A1B12, A2B21, A2B22.
    # Total rank 4 * 7 = 28.
    # For the 2x2x2x1 blocks: A1B13, A2B23.
    # Rank 4 each? No, rank 2 each if we are clever?
    # A(2x2) B(2x1) -> C(2x1).
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Rank 4 is safe. Total 28 + 4 + 4 = 36?
    # Wait, earlier I said 32.
    # 28 + 4 (for C3) = 32.
    # C3 = A1 B13 + A2 B23.
    # A1 B13 is 2x1. A2 B23 is 2x1.
    # We can compute A1 B13 with rank 4 and A2 B23 with rank 4.
    # But they are added.
    # Can we share?
    # No, different A and B.
    # So rank 36?
    # Let's check rank of 2x2x2x1.
    # T(i,j,k) for i=0,1; j=0,1; k=0.
    # T(0,0,0)=1, T(0,1,0)=1, T(1,0,0)=1, T(1,1,0)=1? No.
    # C_i = sum_j A_ij B_j.
    # T(i,j,0) = 1.
    # Rank of this tensor?
    # It is the sum of 4 rank-1 tensors:
    # e_i e_j^T e_0^T.
    # They are linearly independent.
    # So rank is 4.
    # So total rank 28 + 4 + 4 = 36.
    
    # Is 36 better than 40? Yes.
    # Can we do better?
    # Maybe 2x2x2x1 has rank 2?
    # No, 4 terms.
    
    # Let's try to implement rank 36.
    
    def add_strassen_block(U, V, W, rank_offset, A_start, B_start, C_start):
        """
        Add a Strassen 2x2x2x2 block.
        A_start: (i, j) offset in U. A is 2x2.
        B_start: (j, k) offset in V. B is 2x2.
        C_start: (k, i) offset in W. C is 2x2.
        """
        # Strassen factors
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices in local 2x2 block:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # We need to map these to global indices.
        # U index: A_i * 2 + j? No, U is (n*m, rank).
        # A is 2x2. A_start is (i, j) in global A.
        # So U index for A[i,j] is (A_start[0] + i) * m + (A_start[1] + j).
        
        # V index: B[j,k].
        # V index for B[j,k] is (B_start[0] + j) * p + (B_start[1] + k).
        
        # W index: C[k,i].
        # W index for C[k,i] is (C_start[0] + k) * n + (C_start[1] + i).
        
        factors = [
            # (U_indices, V_indices, W_indices)
            # M1
            ([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)]), # (A11+A22)(B11+B22) -> C11, C22
            # M2
            ([(1,0), (1,1)], [(0,0)], [(0,0), (1,0)]), # (A21+A22)B11 -> C21
            # M3
            ([(0,0)], [(1,0), (1,1)], [(0,1), (1,1)]), # A11(B21-B22) -> C12, C22? No.
            # Let's use explicit coefficients.
        ]
        
        # It's easier to just list the 7 products and their contributions to C.
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 - P4 + P5 - P7
        # C01 = P3 + P7
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We will add 7 rank terms.
        for c in range(7):
            col = rank_offset + c
            # Determine U, V, W vectors for this product.
            # This is complex to code generically.
            # Let's hardcode the 7 products for one block.
            
            pass

    # Given the complexity and time limit, I will return the schoolbook solution 
    # but with a simple check if a better one is found via random search?
    # No, the inspiration says Strassen works.
    # I will implement the rank 36 solution explicitly.
    
    # Rank 36 Construction
    U_36 = np.zeros((n * m, 36))
    V_36 = np.zeros((m * p, 36))
    W_36 = np.zeros((p * n, 36))
    
    def add_strassen(U, V, W, offset, a_i, a_j, b_j, b_k, c_k, c_i):
        # a_i, a_j: top-left of A block (2x2)
        # b_j, b_k: top-left of B block (2x2)
        # c_k, c_i: top-left of C block (2x2)
        
        # P1 = (A00+A11)(B00+B11)
        U[offset+0, (a_i+0)*m + (a_j+0)] = 1
        U[offset+0, (a_i+1)*m + (a_j+1)] = 1
        V[offset+0, (b_j+0)*p + (b_k+0)] = 1
        V[offset+0, (b_j+1)*p + (b_k+1)] = 1
        W[offset+0, (c_k+0)*n + (c_i+0)] = 1
        W[offset+0, (c_k+1)*n + (c_i+1)] = 1
        
        # P2 = (A10+A11)B00
        U[offset+1, (a_i+1)*m + (a_j+0)] = 1
        U[offset+1, (a_i+1)*m + (a_j+1)] = 1
        V[offset+1, (b_j+0)*p + (b_k+0)] = 1
        W[offset+1, (c_k+0)*n + (c_i+0)] = 1
        W[offset+1, (c_k+1)*n + (c_i+0)] = 1 # Wait, C10 gets P2.
        # C10 is (1,0) in C block.
        # So W index for C(1,0) is (c_k+0)*n + (c_i+1).
        W[offset+1, (c_k+0)*n + (c_i+1)] = 1
        W[offset+1, (c_k+1)*n + (c_i+1)] = 0 # Correction: P2 contributes to C10 only?
        # C10 = P2 + P4.
        # So P2 contributes to C(1,0).
        # W[offset+1, (c_k+0)*n + (c_i+1)] = 1.
        # W[offset+1, (c_k+1)*n + (c_i+1)] = 0.
        # Let's restart W assignment for P2.
        W[offset+1, :] = 0
        W[offset+1, (c_k+0)*n + (c_i+1)] = 1
        
        # P3 = A00(B10-B11)
        U[offset+2, (a_i+0)*m + (a_j+0)] = 1
        V[offset+2, (b_j+1)*p + (b_k+0)] = 1
        V[offset+2, (b_j+1)*p + (b_k+1)] = -1
        W[offset+2, (c_k+0)*n + (c_i+1)] = 1 # C01
        W[offset+2, (c_k+1)*n + (c_i+1)] = 1 # C11? No.
        # C01 = P3 + P7.
        # C11 = P1 - P2 + P3 + P6.
        # So P3 contributes to C01 and C11.
        W[offset+2, :] = 0
        W[offset+2, (c_k+0)*n + (c_i+1)] = 1
        W[offset+2, (c_k+1)*n + (c_i+1)] = 1
        
        # P4 = A11(B01-B00)
        U[offset+3, (a_i+1)*m + (a_j+1)] = 1
        V[offset+3, (b_j+0)*p + (b_k+1)] = 1
        V[offset+3, (b_j+0)*p + (b_k+0)] = -1
        # C00 = P1 - P4 + P5 - P7.
        # C10 = P2 + P4.
        W[offset+3, :] = 0
        W[offset+3, (c_k+0)*n + (c_i+0)] = -1
        W[offset+3, (c_k+1)*n + (c_i+1)] = 1
        
        # P5 = (A00+A01)B11
        U[offset+4, (a_i+0)*m + (a_j+0)] = 1
        U[offset+4, (a_i+0)*m + (a_j+1)] = 1
        V[offset+4, (b_j+1)*p + (b_k+1)] = 1
        # C00 = P1 - P4 + P5 - P7.
        W[offset+4, :] = 0
        W[offset+4, (c_k+0)*n + (c_i+0)] = 1
        
        # P6 = (A10-A00)(B00+B01)
        U[offset+5, (a_i+1)*m + (a_j+0)] = 1
        U[offset+5, (a_i+0)*m + (a_j+0)] = -1
        V[offset+5, (b_j+0)*p + (b_k+0)] = 1
        V[offset+5, (b_j+0)*p + (b_k+1)] = 1
        # C11 = P1 - P2 + P3 + P6.
        W[offset+5, :] = 0
        W[offset+5, (c_k+1)*n + (c_i+1)] = 1
        
        # P7 = (A01-A11)(B10+B11)
        U[offset+6, (a_i+0)*m + (a_j+1)] = 1
        U[offset+6, (a_i+1)*m + (a_j+1)] = -1
        V[offset+6, (b_j+1)*p + (b_k+0)] = 1
        V[offset+6, (b_j+1)*p + (b_k+1)] = 1
        # C00 = P1 - P4 + P5 - P7.
        # C01 = P3 + P7.
        W[offset+6, :] = 0
        W[offset+6, (c_k+0)*n + (c_i+0)] = -1
        W[offset+6, (c_k+0)*n + (c_i+1)] = 1

    # Block 1: A1 B11 -> C[:, 0:2] part 1
    # A1 is A[0:2, 0:2] -> a_i=0, a_j=0
    # B11 is B[0:2, 0:2] -> b_j=0, b_k=0
    # C part is C[0:2, 0:2] -> c_k=0, c_i=0
    add_strassen(U_36, V_36, W_36, 0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A1 B12 -> C[:, 2:4] part 1
    # A1 is A[0:2, 0:2]
    # B12 is B[0:2, 2:4] -> b_j=0, b_k=2
    # C part is C[0:2, 2:4] -> c_k=2, c_i=0
    add_strassen(U_36, V_36, W_36, 7, 0, 0, 0, 2, 2, 0)
    
    # Block 3: A2 B21 -> C[:, 0:2] part 2
    # A2 is A[0:2, 2:4] -> a_i=0, a_j=2
    # B21 is B[2:4, 0:2] -> b_j=2, b_k=0
    # C part is C[0:2, 0:2] -> c_k=0, c_i=0
    add_strassen(U_36, V_36, W_36, 14, 0, 2, 2, 0, 0, 0)
    
    # Block 4: A2 B22 -> C[:, 2:4] part 2
    # A2 is A[0:2, 2:4]
    # B22 is B[2:4, 2:4] -> b_j=2, b_k=2
    # C part is C[0:2, 2:4] -> c_k=2, c_i=0
    add_strassen(U_36, V_36, W_36, 21, 0, 2, 2, 2, 2, 0)
    
    # Block 5: A1 B13 -> C[:, 4:5] part 1
    # A1 is A[0:2, 0:2]
    # B13 is B[0:2, 4:5] -> b_j=0, b_k=4 (2x1)
    # C part is C[0:2, 4:5] -> c_k=4, c_i=0 (2x1)
    # Rank 4 for 2x2x2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # Term 1: A00 B0
    U_36[28, 0*m + 0] = 1
    V_36[28, 0*p + 4] = 1
    W_36[28, 4*n + 0] = 1
    
    # Term 2: A01 B1
    U_36[29, 0*m + 1] = 1
    V_36[29, 1*p + 4] = 1
    W_36[29, 4*n + 0] = 1
    
    # Term 3: A10 B0
    U_36[30, 1*m + 0] = 1
    V_36[30, 0*p + 4] = 1
    W_36[30, 4*n + 1] = 1
    
    # Term 4: A11 B1
    U_36[31, 1*m + 1] = 1
    V_36[31, 1*p + 4] = 1
    W_36[31, 4*n + 1] = 1
    
    # Block 6: A2 B23 -> C[:, 4:5] part 2
    # A2 is A[0:2, 2:4]
    # B23 is B[2:4, 4:5] -> b_j=2, b_k=4
    # C part is C[0:2, 4:5] -> c_k=4, c_i=0
    
    # Term 5: A200 B20
    U_36[32, 0*m + 2] = 1
    V_36[32, 2*p + 4] = 1
    W_36[32, 4*n + 0] = 1
    
    # Term 6: A201 B21
    U_36[33, 0*m + 3] = 1
    V_36[33, 3*p + 4] = 1
    W_36[33, 4*n + 0] = 1
    
    # Term 7: A210 B20
    U_36[34, 1*m + 2] = 1
    V_36[34, 2*p + 4] = 1
    W_36[34, 4*n + 1] = 1
    
    # Term 8: A211 B21
    U_36[35, 1*m + 3] = 1
    V_36[35, 3*p + 4] = 1
    W_36[35, 4*n + 1] = 1
    
    # Verify
    def verify_decomp_fast(U, V, W, tol=1e-6):
        rank = U.shape[1]
        U_idx = U.reshape(n, m, rank)
        V_idx = V.reshape(m, p, rank)
        W_idx = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        if not np.allclose(T, np.ones((n, m, p)), atol=tol):
            return False
        return True

    if verify_decomp_fast(U_36, V_36, W_36):
        best_rank = 36
        best_U = U_36
        best_V = V_36
        best_W = W_36

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
