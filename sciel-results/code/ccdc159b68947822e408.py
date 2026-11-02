"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from collections import defaultdict
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Try to find a rank-48 decomposition for 4x4 matrix multiplication
    # using the BCLR (Bini-Capovani-Lotti-Romani) construction.
    # Fallback to Strassen (rank 49) or schoolbook (rank 64) if not found.
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Keep 10s margin
    
    # First, try to construct a valid rank-48 decomposition
    # The BCLR decomposition for 4x4 has rank 48
    # We'll use a known explicit decomposition
    
    # For 4x4 matrix multiplication, we can use the following approach:
    # Split 4x4 into 2x2 blocks and use Strassen recursively
    # But BCLR improves Strassen from 49 to 48
    
    # Let's implement a direct rank-48 decomposition
    # The decomposition uses 48 bilinear forms
    
    # Strategy: Use the explicit BCLR decomposition
    # This is a known result in tensor decomposition theory
    
    # Build U, V, W matrices with 48 columns
    rank = 48
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # The BCLR 48-term decomposition for 4x4 matrix multiplication
    # Each term corresponds to a triple (i, j, k) pattern with coefficients
    
    # We'll generate the decomposition using a systematic approach
    # Based on the known BCLR construction
    
    # For 4x4, we can view the matrices as 2x2 blocks of 2x2 submatrices
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2
    
    # Each Cij = sum of bilinear forms in the 2x2 blocks
    
    # Using Strassen for the block level gives 7 forms for the 2x2 block mult
    # But we need to handle the 2x2 submatrix multiplications too
    
    # Let me use a different approach: explicit known decomposition
    
    # The BCLR decomposition for M(4,4,4) has rank 48
    # I'll use a parameterized family and search for valid coefficients
    
    # Alternative: Use a greedy/optimization approach
    # Start with Strassen-like structure and optimize
    
    # For robustness, let's first verify if we can achieve rank < 64
    # by using a structured decomposition
    
    # Build a rank-49 Strassen decomposition as fallback
    def strassen_4x4():
        # 4x4 via 2x2 Strassen blocks
        # Each 2x2 block mult uses 7 Strassen forms
        # Total: we need to combine properly
        
        # Let's use the standard approach:
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # where each is 2x2
        
        # M1 = (A21 + A22) * B12
        # M2 = A11 * (B21 - B11)  
        # M3 = (A11 + A12) * B22
        # M4 = A22 * (B12 - B22)
        # M5 = (A11 + A22) * (B11 + B22)
        # M6 = (A12 - A22) * (B21 + B22)
        # M7 = (A21 - A11) * (B11 + B12)
        
        # C11 = M5 + M4 - M3 + M1
        # C12 = M3 + M6
        # C21 = M7 + M2
        # C22 = M1 + M5 - M2 + M4
        
        # Each Mk is a product of two 2x2 matrices
        # So each Mk = X * Y where X, Y are 2x2
        # This gives 4 entries per Mk, total 7*4 = 28 bilinear forms for the block level
        # But we also need to account for the 2x2 submatrix multiplications
        
        # Actually, let me think differently.
        
        # For 4x4, the tensor has 4*4*4 = 64 elements
        # A rank-r decomposition has r trilinear terms
        
        # Let me just use a direct construction
        # I'll use the fact that 4x4 can be decomposed into 4 independent 2x2 mults
        # plus cross terms, but that's not optimal
        
        # Better: use the known BCLR result directly
        # The BCLR 48-term decomposition exists and is well-documented
        
        # For now, let's return a valid rank-64 schoolbook decomposition
        # and try to improve it
        
        return None
    
    # Let me try a different strategy: use random initialization and 
    # gradient-based optimization to minimize the residual
    
    # Actually, let me just implement a known explicit decomposition
    # The BCLR decomposition for 4x4 matrix multiplication
    
    # Here's a systematic approach:
    # We know that M(4) = 48 (BCLR 1979)
    # Let's construct it explicitly
    
    # The decomposition can be written as:
    # C = sum_{s=1}^{48} U_s * V_s * W_s (outer products)
    
    # Where U_s, V_s are 4x4 matrices (flattened to 16-vectors)
    # and W_s is a 4x4 matrix
    
    # Let me use a known parametrization
    
    # For 4x4, we can use the following block structure:
    # Split indices into high and low bits
    # i = 2*i2 + i1, j = 2*j2 + j1, k = 2*k2 + k1
    # where i1, i2, j1, j2, k1, k2 in {0, 1}
    
    # This gives a 2x2 block structure at two levels
    
    # Level 1: 2x2 blocks of 2x2 submatrices
    # Level 2: within each 2x2 submatrix, 2x2 of 1x1
    
    # Using Strassen at both levels:
    # Level 2 (1x1 mult): rank 2 (Strassen for 2x2 scalar = 2? No, 2x2 is 2x2)
    # Wait, for 1x1, it's just multiplication, rank 1
    
    # Let me reconsider.
    
    # I'll use a numerical approach: initialize with random values,
    # then use alternating least squares or gradient descent
    
    # For robustness and time constraints, let me return the schoolbook
    # decomposition which is guaranteed to work, and add a timeout-guarded
    # attempt to find a better one
    
    # Schoolbook decomposition (rank 64)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Check: for all i, j, k: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = delta_{i,k} * (A_ij * B_jk coefficient)
        # Actually, the matrix multiplication tensor T[i,j,k] = A[i,j] * B[j,k]
        # The decomposition should satisfy:
        # T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # where T[i,j,k] = 1 if i==k (for the identity part), else 0
        # Wait, that's not right.
        
        # The matrix multiplication tensor is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
        # No, that's not right either.
        
        # The tensor for matrix multiplication is:
        # T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # But T[i,j,k] should be 1 if the indices match properly.
        
        # Actually, the standard tensor decomposition for matrix mult is:
        # T is a 3-tensor of order (n*m, m*p, p*n) or similar
        # T[a, b, c] = 1 if a, b, c correspond to (i,j), (j,k), (k,i) with matching j and k
        
        # Let me re-read the task:
        # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # The decomposition is: T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        # where a = i*m+j, b = j*p+k, c = k*n+i
        # T[a,b,c] = 1 if the indices are consistent (i.e., the j in a matches j in b, and k in b matches k in c, and i in a matches i in c)
        # Otherwise 0
        
        # So T[a,b,c] = 1 iff j_a == j_b and k_b == k_c and i_a == i_c
        # where a = i_a * m + j_a, b = j_b * p + k_b, c = k_c * n + i_c
        
        # For the schoolbook: each term has one (i,j,k) triple
        # U[i*m+j, s] = 1, V[j*p+k, s] = 1, W[k*n+i, s] = 1
        # So for this s, T[a,b,c] gets contribution 1 only when a=i*m+j, b=j*p+k, c=k*n+i
        # And T[i*m+j, j*p+k, k*n+i] should be 1 (since indices match)
        # All other entries get 0 from this term
        # So the sum gives T[a,b,c] = 1 iff there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
        # which is exactly when j_a=j_b, k_b=k_c, i_a=i_c
        # So it's correct.
        
        # Let me verify by checking a few entries
        if rank == 0:
            return True
        
        # Compute the reconstructed tensor
        # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        # This is a 3-way tensor product (Khatri-Rao / parallel factor)
        
        # For efficiency, check all entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Expected value
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    expected = 1.0 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0.0
                    
                    # Computed value
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - expected) > 1e-6:
                        return False
        return True
    
    # Check schoolbook
    if verify(U_sb, V_sb, W_sb):
        # Schoolbook is valid with rank 64
        # Try to find a better decomposition
        # For now, return schoolbook as a safe fallback
        pass
    
    # Let me try to implement the BCLR 48-term decomposition
    # This is the optimal rank for 4x4 matrix multiplication
    
    # I'll use a known explicit construction
    # The BCLR decomposition uses 48 terms with coefficients in {0, ±1, ±1/2}
    
    # Due to complexity, let me use a different approach:
    # Use the fact that we can reduce the rank from 64 to 49 using Strassen
    # and then potentially to 48 using BCLR
    
    # For Strassen on 4x4:
    # Treat 4x4 as 2x2 blocks of 2x2
    # Use 7 Strassen forms at the block level
    # Each block mult is 2x2, which itself can be done with 7 Strassen forms
    # But that would give 7*7 = 49 forms? No, that's not how it works.
    
    # Actually, for 4x4 via 2x2 blocks:
    # We have 4 block multiplications: C11 = A11*B11 + A12*B21, etc.
    # Strassen reduces the 2x2 block mult to 7 forms
    # Each form is a product of two 2x2 matrices
    # Each 2x2 product has 4 entries
    # So total bilinear forms: 7 * 4 = 28? No...
    
    # Wait, I'm confusing myself. Let me think clearly.
    
    # The tensor for 4x4 matrix mult has size 16 * 16 * 16 = 4096 entries
    # A rank-r decomposition has r terms, each term is a product of three vectors
    # of lengths 16, 16, 16
    
    # For Strassen on 2x2: rank 7
    # For 4x4 via 2x2 Strassen blocks:
    # The 4x4 mult can be done in 49 operations using recursive Strassen
    # (2^3 = 8? No, 4x4 = 2*(2x2), Strassen gives 7 instead of 8, so 7^2 = 49)
    
    # So rank 49 is achievable via recursive Strassen
    
    # BCLR improves to 48
    
    # Let me implement recursive Strassen for rank 49
    
    def strassen_4x4_decomp():
        """Construct a rank-49 decomposition using recursive Strassen."""
        # 4x4 via 2x2 Strassen blocks
        # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
        # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
        
        # Strassen forms for the 2x2 block mult:
        # M1 = (A21 + A22) * B12
        # M2 = A11 * (B21 - B11)
        # M3 = (A11 + A12) * B22
        # M4 = A22 * (B12 - B22)
        # M5 = (A11 + A22) * (B11 + B22)
        # M6 = (A12 - A22) * (B21 + B22)
        # M7 = (A21 - A11) * (B11 + B12)
        
        # C11 = M5 + M4 - M3 + M1
        # C12 = M3 + M6
        # C21 = M7 + M2
        # C22 = M1 + M5 - M2 + M4
        
        # Each Mk is a 2x2 matrix product
        # Mk = Xk * Yk where Xk, Yk are 2x2 matrices
        # The product Xk*Yk has 4 entries, each is a bilinear form in the entries of Xk and Yk
        
        # So each Mk contributes 4 bilinear forms
        # Total: 7 * 4 = 28 bilinear forms at the block level
        
        # But wait, that's not right. Each Mk is a single bilinear form at the block level,
        # but when we expand to scalar level, it becomes 4 scalar bilinear forms.
        
        # Hmm, let me think again.
        
        # The tensor for 2x2 matrix mult has rank 7
        # This means we can write:
        # (C11, C12, C21, C22) = sum_{s=1}^{7} u_s(A) * v_s(B) * w_s(C)
        # where u_s, v_s are linear forms in A, B entries and w_s is a linear form in C entries
        
        # For 4x4, if we treat it as 2x2 blocks, the tensor structure is:
        # T_4x4 = T_2x2_block ⊗ T_2x2_scalar (roughly)
        # The rank is multiplicative in some sense, but not exactly
        
        # Actually, the rank of the tensor for n x n matrix mult satisfies:
        # M(2n) <= M(n)^2
        # So M(4) <= M(2)^2 = 7^2 = 49
        
        # To achieve rank 49, we can use the tensor product of two Strassen decompositions
        
        # Let me construct it explicitly
        
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Index mapping:
        # a = i*m + j = 4*i + j, where i in {0,1,2,3}, j in {0,1,2,3}
        # We can write i = 2*i2 + i1, j = 2*j2 + j1
        # So a = 4*(2*i2+i1) + (2*j2+j1) = 8*i2 + 4*i1 + 2*j2 + j1
        
        # Similarly for b and c
        
        # The 4x4 mult tensor can be viewed as a 2x2 block tensor
        # where each block is a 2x2 scalar tensor
        
        # Using Strassen at both levels:
        # Level 1 (block): 7 forms
        # Level 2 (scalar within block): 7 forms
        # Total: 7 * 7 = 49 forms
        
        # For each pair (s1, s2) where s1 in {1..7} (block Strassen) and s2 in {1..7} (scalar Strassen):
        # The form is: U[a, s] * V[b, s] * W[c, s]
        # where s = s1*7 + s2 - 1
        
        # Let me define the Strassen forms for 2x2:
        # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
        # C = [[c0, c1], [c2, c3]]
        
        # M1 = (a2+a3)*b1
        # M2 = a0*(b2-b0)
        # M3 = (a0+a1)*b3
        # M4 = a3*(b1-b3)
        # M5 = (a0+a3)*(b0+b3)
        # M6 = (a1-a3)*(b2+b3)
        # M7 = (a2-a0)*(b0+b1)
        
        # C0 = M5 + M4 - M3 + M1
        # C1 = M3 + M6
        # C2 = M7 + M2
        # C3 = M1 + M5 - M2 + M4
        
        # For the block level, similar but with 2x2 matrices instead of scalars
        
        # This is getting complex. Let me use a different approach.
        
        # I'll use the fact that the tensor for 4x4 mult is the Kronecker product
        # of the tensor for 2x2 mult with itself (up to permutation)
        
        # The rank of a Kronecker product of tensors is the product of ranks
        # So M(4) = M(2) * M(2) = 7 * 7 = 49
        
        # Let me construct it using the Kronecker product structure
        
        # Define the Strassen decomposition for 2x2:
        # T_2x2[a,b,c] = sum_{s=1}^{7} U2[a,s] * V2[b,s] * W2[c,s]
        # where a, b, c in {0,1,2,3} (flattened 2x2 indices)
        
        # U2, V2, W2 are 4x7 matrices
        
        # For 4x4:
        # a = a1 * 2 + a0, b = b1 * 2 + b0, c = c1 * 2 + c0
        # where a1, a0, b1, b0, c1, c0 in {0, 1}
        
        # T_4x4[a,b,c] = sum_{s1,s2} U2[a1,s1] * V2[b1,s1] * W2[c1,s1] * U2[a0,s2] * V2[b0,s2] * W2[c0,s2]
        
        # This can be rewritten as:
        # T_4x4[a,b,c] = sum_{s} U4[a,s] * V4[b,s] * W4[c,s]
        # where s = s1*4 + s0 (using 4 for the scalar level? No, 7)
        # s = s1*7 + s2
        
        # U4[a, s1*7+s2] = U2[a1, s1] * U2[a0, s2]
        # Similarly for V4 and W4
        
        # Let me implement this
        
        # First, define U2, V2, W2 for 2x2 Strassen
        
        # 2x2 index mapping: idx = 2*i + j, i,j in {0,1}
        # A[idx] = A[i,j], B[idx] = B[i,j], C[idx] = C[i,j]
        
        # Strassen forms:
        # Let me define the coefficient vectors
        
        # For U2 (input A):
        # M1: A21 + A22 -> indices: A[2*1+1]=A[3]? No, A[i,j] with i=1,j=1 -> A[3], A[1,j]=A[1], A[1,0]=A[2]
        # Wait, A = [[A00, A01], [A10, A11]], so A[i,j] = A[2*i+j]
        # A00 = A[0], A01 = A[1], A10 = A[2], A11 = A[3]
        
        # M1 = (A10 + A11) * B01 = (A[2]+A[3]) * B[1]
        # M2 = A00 * (B10 - B00) = A[0] * (B[2]-B[0])
        # M3 = (A00 + A01) * B11 = (A[0]+A[1]) * B[3]
        # M4 = A11 * (B01 - B11) = A[3] * (B[1]-B[3])
        # M5 = (A00 + A11) * (B00 + B11) = (A[0]+A[3]) * (B[0]+B[3])
        # M6 = (A01 - A11) * (B10 + B11) = (A[1]-A[3]) * (B[2]+B[3])
        # M7 = (A10 - A00) * (B00 + B01) = (A[2]-A[0]) * (B[0]+B[1])
        
        # C00 = M5 + M4 - M3 + M1
        # C01 = M3 + M6
        # C10 = M7 + M2
        # C11 = M1 + M5 - M2 + M4
        
        # So W2[c, s]:
        # c=0 (C00): W2[0, s] = [1, 0, -1, 1, 1, 0, 0] for s=1..7 (M1, M2, M3, M4, M5, M6, M7)
        # Wait, M1 contributes +1, M4 contributes +1, M3 contributes -1, M5 contributes +1
        # So W2[0, :] = [1, 0, -1, 1, 1, 0, 0]
        
        # c=1 (C01): M3 + M6 -> [0, 0, 1, 0, 0, 1, 0]
        # c=2 (C10): M7 + M2 -> [0, 1, 0, 0, 0, 0, 1]
        # c=3 (C11): M1 + M5 - M2 + M4 -> [1, -1, 0, 1, 1, 0, 0]
        
        # U2[a, s]:
        # M1: A2+A3 -> [0, 0, 1, 1]
        # M2: A0 -> [1, 0, 0, 0]
        # M3: A0+A1 -> [1, 1, 0, 0]
        # M4: A3 -> [0, 0, 0, 1]
        # M5: A0+A3 -> [1, 0, 0, 1]
        # M6: A1-A3 -> [0, 1, 0, -1]
        # M7: A2-A0 -> [-1, 0, 1, 0]
        
        # So U2 = [[0, 1, 1, 0, 1, 0, -1],
        #           [0, 0, 1, 0, 0, 1, 0],
        #           [1, 0, 0, 0, 0, 0, 1],
        #           [1, 0, 0, 1, 1, -1, 0]]
        
        # V2[b, s]:
        # M1: B1 -> [0, 1, 0, 0]
        # M2: B2-B0 -> [-1, 0, 1, 0]
        # M3: B3 -> [0, 0, 0, 1]
        # M4: B1-B3 -> [0, 1, 0, -1]
        # M5: B0+B3 -> [1, 0, 0, 1]
        # M6: B2+B3 -> [0, 0, 1, 1]
        # M7: B0+B1 -> [1, 1, 0, 0]
        
        # So V2 = [[-1, 0, 0, 1, 1, 0, 1],
        #           [1, 0, 0, 1, 0, 0, 1],
        #           [0, 1, 0, 0, 0, 1, 0],
        #           [0, 0, 1, -1, 1, 1, 0]]
        
        U2 = np.array([
            [0, 1, 1, 0, 1, 0, -1],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, -1, 0]
        ], dtype=float)
        
        V2 = np.array([
            [-1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, -1, 1, 1, 0]
        ], dtype=float)
        
        W2 = np.array([
            [1, 0, -1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [1, -1, 0, 1, 1, 0, 0]
        ], dtype=float)
        
        # Verify 2x2 Strassen
        def verify_2x2(U2, V2, W2):
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        i_a, j_a = a // 2, a % 2
                        j_b, k_b = b // 2, b % 2
                        k_c, i_c = c // 2, c % 2
                        expected = 1.0 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0.0
                        val = np.dot(U2[a, :], V2[b, :] * W2[c, :])
                        if abs(val - expected) > 1e-6:
                            return False
            return True
        
        # Now construct 4x4 from 2x2 using Kronecker product
        # a = 4*i + j = 4*(2*i2+i1) + (2*j2+j1) = 8*i2 + 4*i1 + 2*j2 + j1
        # So a1 = a // 2, a0 = a % 2 (high and low bits)
        # Similarly for b, c
        
        rank = 49
        U4 = np.zeros((16, rank), dtype=float)
        V4 = np.zeros((16, rank), dtype=float)
        W4 = np.zeros((16, rank), dtype=float)
        
        for s1 in range(7):
            for s2 in range(7):
                s = s1 * 7 + s2
                for a in range(16):
                    a1 = a // 2
                    a0 = a % 2
                    U4[a, s] = U2[a1, s1] * U2[a0, s2]
                for b in range(16):
                    b1 = b // 2
                    b0 = b % 2
                    V4[b, s] = V2[b1, s1] * V2[b0, s2]
                for c in range(16):
                    c1 = c // 2
                    c0 = c % 2
                    W4[c, s] = W2[c1, s1] * W2[c0, s2]
        
        # Verify
        if verify(U4, V4, W4):
            return U4, V4, W4
        else:
            return None
    
    # Try to get Strassen 4x4 decomposition
    result = strassen_4x4_decomp()
    if result is not None:
        U, V, W = result
        # Return as list of lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

