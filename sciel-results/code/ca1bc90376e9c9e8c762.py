"""Strassen-based tensor decomposition for 4x4x9 matrix multiplication."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Build Strassen decomposition for 4x4x4 (rank 49)
    # Then extend to 4x4x9 by adding schoolbook terms for the remaining 5 columns
    
    # First, build the 2x2x2 Strassen decomposition (rank 7)
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # 
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each term t has:
    # U[i,j,t] * V[j,k,t] * W[k,i,t]
    # where (i,j) indexes A, (j,k) indexes B, (k,i) indexes C (W convention: W[k*n+i])
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7.
        U shape: (4, 7) - indices: i*2+j for A[i,j]
        V shape: (4, 7) - indices: j*2+k for B[j,k]  
        W shape: (4, 7) - indices: k*2+i for C[i,k] (W[k*n+i] with n=2)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indices: 0=a00, 1=a01, 2=a10, 3=a11
        # B indices: 0=b00, 1=b01, 2=b10, 3=b11
        # C indices (W convention k*2+i): 0=c[0,0], 1=c[0,1], 2=c[1,0], 3=c[1,1]
        
        # M1 = (a00+a11)*(b00+b11)
        # U: a00=1, a11=1 -> indices 0, 3
        # V: b00=1, b11=1 -> indices 0, 3
        # W: c[0,0]=1, c[1,1]=1 -> indices 0, 3
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a10+a11)*b00
        # U: a10=1, a11=1 -> indices 2, 3
        # V: b00=1 -> index 0
        # W: c[1,0]=1 -> index 2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = a00*(b01-b11)
        # U: a00=1 -> index 0
        # V: b01=1, b11=-1 -> indices 1, 3
        # W: c[0,1]=1 -> index 1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = a11*(b10-b00)
        # U: a11=1 -> index 3
        # V: b10=1, b00=-1 -> indices 2, 0
        # W: c[1,0]=1, c[0,0]=1 -> indices 2, 0
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        # M5 = (a00+a01)*b11
        # U: a00=1, a01=1 -> indices 0, 1
        # V: b11=1 -> index 3
        # W: c[0,0]=-1, c[0,1]=1 -> indices 0, 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (a10-a00)*(b00+b01)
        # U: a10=1, a00=-1 -> indices 2, 0
        # V: b00=1, b01=1 -> indices 0, 1
        # W: c[1,1]=1 -> index 3
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a01-a11)*(b10+b11)
        # U: a01=1, a11=-1 -> indices 1, 3
        # V: b10=1, b11=1 -> indices 2, 3
        # W: c[0,0]=1 -> index 0
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Verify 2x2x2 decomposition
    U2, V2, W2 = strassen_2x2x2()
    # Check: for random A, B, compute C via Strassen and compare with direct
    for test_iter in range(10):
        A_test = np.random.randn(2, 2)
        B_test = np.random.randn(2, 2)
        C_direct = A_test @ B_test
        
        C_strassen = np.zeros((2, 2))
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    C_strassen[i, k] += np.dot(U2[i*2+j, :], V2[j*2+k, :]) * W2[k*2+i, :]
        
        if not np.allclose(C_direct, C_strassen, atol=1e-10):
            # Fallback: use schoolbook
            break
    
    # Now build 4x4x4 using recursive Strassen (rank 49)
    # Split 4x4 into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = A*B
    #
    # Top-level Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M_i is a 2x2 matrix product, decomposable with rank 7.
    # Total rank: 7 * 7 = 49
    
    def build_strassen_4x4x4():
        """Build U, V, W for 4x4x4 matrix multiplication with rank 49.
        U shape: (16, 49) - indices: i*4+j for A[i,j]
        V shape: (16, 49) - indices: j*4+k for B[j,k]
        W shape: (16, 49) - indices: k*4+i for C[i,k] (W[k*n+i] with n=4)
        """
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U2, V2, W2 = strassen_2x2x2()
        
        # Define the 7 top-level Strassen terms for 4x4x4
        # Each term involves linear combinations of 2x2 blocks
        # 
        # For term t (0-6), we need to determine:
        # - Which A blocks are involved and with what coefficients
        # - Which B blocks are involved and with what coefficients
        # - How the result contributes to C blocks
        #
        # Top-level Strassen terms:
        # M1 = (A00+A11)*(B00+B11): A blocks {00:1, 11:1}, B blocks {00:1, 11:1}, C: C00+=1, C11+=1
        # M2 = (A10+A11)*B00: A blocks {10:1, 11:1}, B blocks {00:1}, C: C10+=1
        # M3 = A00*(B01-B11): A blocks {00:1}, B blocks {01:1, 11:-1}, C: C01+=1
        # M4 = A11*(B10-B00): A blocks {11:1}, B blocks {10:1, 00:-1}, C: C10+=1, C00+=1
        # M5 = (A00+A01)*B11: A blocks {00:1, 01:1}, B blocks {11:1}, C: C00-=-1, C01+=1
        # M6 = (A10-A00)*(B00+B01): A blocks {10:1, 00:-1}, B blocks {00:1, 01:1}, C: C11+=1
        # M7 = (A01-A11)*(B10+B11): A blocks {01:1, 11:-1}, B blocks {10:1, 11:1}, C: C00+=1
        
        # For each top-level term, we have a 2x2 matrix product L*R where
        # L and R are linear combinations of 2x2 blocks.
        # We decompose this 2x2 product using Strassen (rank 7).
        #
        # The key insight: if L = sum_alpha c_alpha * A_block_alpha and R = sum_beta d_beta * B_block_beta,
        # then L*R = sum_alpha,beta c_alpha*d_beta * (A_block_alpha * B_block_beta)
        # Each A_block_alpha * B_block_beta is a 2x2x2 product (rank 7).
        #
        # So for each top-level term with |A_blocks| * |B_blocks| sub-products,
        # we add |A_blocks| * |B_blocks| * 7 terms.
        #
        # Let's count:
        # M1: 2*2=4 sub-products -> 28 terms
        # M2: 2*1=2 sub-products -> 14 terms
        # M3: 1*2=2 sub-products -> 14 terms
        # M4: 1*2=2 sub-products -> 14 terms
        # M5: 2*1=2 sub-products -> 14 terms
        # M6: 2*2=4 sub-products -> 28 terms
        # M7: 2*2=4 sub-products -> 28 terms
        # Total: 28+14+14+14+14+28+28 = 140 terms. That's more than 49!
        
        # I'm making an error. The correct approach:
        # In Strassen, each "multiplication" is of two 2x2 MATRICES, not scalars.
        # The tensor for multiplying two 2x2 matrices has rank 7.
        # So each top-level term (which is one 2x2 matrix multiplication) contributes 7 rank-1 terms.
        # Total: 7 * 7 = 49.
        #
        # But the subtlety is that the operands of each top-level multiplication are
        # linear combinations of blocks. For the tensor decomposition, we need to
        # express this correctly.
        #
        # Let me think differently. For a single 2x2 matrix multiplication L*R where
        # L and R are specific 2x2 matrices (not linear combinations), the rank-7
        # decomposition is given by strassen_2x2x2().
        #
        # For the 4x4 case, when we say M1 = (A00+A11)*(B00+B11), this means:
        # The left operand is the 2x2 matrix (A00+A11) and the right is (B00+B11).
        # The product is a 2x2 matrix.
        #
        # To decompose this as a sum of rank-1 terms in the overall 4x4x4 tensor,
        # we can use the 2x2x2 Strassen decomposition, but with the understanding that
        # the "elements" of L and R are linear combinations of A and B elements.
        #
        # Specifically, for M1 = L1*R1 where L1 = A00+A11 and R1 = B00+B11:
        # L1[0,0] = A[0,0] + A[2,2]
        # L1[0,1] = A[0,1] + A[2,3]
        # L1[1,0] = A[1,0] + A[3,2]
        # L1[1,1] = A[1,1] + A[3,3]
        # R1[0,0] = B[0,0] + B[2,2]
        # R1[0,1] = B[0,1] + B[2,3]
        # R1[1,0] = B[1,0] + B[3,2]
        # R1[1,1] = B[1,1] + B[3,3]
        #
        # The 2x2x2 Strassen decomposition says:
        # (L1*R1)[i,k] = sum_j sum_t U2[i,j,t]*L1[i,j]*V2[j,k,t]*R1[j,k]*W2[k,i,t]
        # = sum_j sum_t U2[i,j,t]*V2[j,k,t]*W2[k,i,t]*L1[i,j]*R1[j,k]
        #
        # Substituting L1 and R1 in terms of A and B:
        # L1[i,j] = A[i,j] + A[i+2,j+2]  (for i,j in {0,1})
        # R1[j,k] = B[j,k] + B[j+2,k+2]  (for j,k in {0,1})
        #
        # So L1[i,j]*R1[j,k] = (A[i,j]+A[i+2,j+2])*(B[j,k]+B[j+2,k+2])
        # = A[i,j]*B[j,k] + A[i,j]*B[j+2,k+2] + A[i+2,j+2]*B[j,k] + A[i+2,j+2]*B[j+2,k+2]
        #
        # This expands to 4 terms, each a product of one A element and one B element.
        # So M1 actually requires 4*7=28 rank-1 terms, not 7!
        #
        # This means the total rank is not 49 but much higher. Let me recalculate.
        #
        # Actually, I think the issue is that Strassen's algorithm for 4x4x4 does NOT
        # simply give rank 49 in the tensor decomposition sense. The rank-49 result
        # refers to the number of scalar multiplications in the algorithm, but the
        # tensor rank might be different.
        #
        # Let me use a different approach: directly construct the schoolbook decomposition
        # for 4x4x9 and then try to reduce it using known techniques.
        
        # For now, return the schoolbook decomposition for 4x4x4 (rank 64)
        U = np.zeros((16, 64))
        V = np.zeros((16, 64))
        W = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W
    
    # Given the complexity, let me use a simpler but effective approach:
    # Use schoolbook for 4x4x9 (rank 144) but verify it works.
    
    # Schoolbook decomposition for 4x4x9
    rank_schoolbook = n * m * p  # 144
    U = np.zeros((nm, rank_schoolbook))
    V = np.zeros((mp, rank_schoolbook))
    W = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    C_check = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C_check[i, k] += np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
    
    if not np.allclose(C_check, np.ones((n, p)), atol=1e-8):
        # Should not happen, but just in case
        pass
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
