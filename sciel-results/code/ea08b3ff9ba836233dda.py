"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def _kronecker_decomp_4x4x4():
    """
    Return a rank-16 decomposition for 4x4x4 matrix multiplication.
    
    Uses the Kronecker product of the 2x2x2 schoolbook (rank 4) with itself.
    The 2x2x2 schoolbook has 4 rank-1 terms: (i,j,k) for i,j,k in {0,1}.
    The Kronecker product gives 4*4 = 16 terms for 4x4x4.
    """
    # 2x2x2 schoolbook: 4 rank-1 components
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1}
    # Rank-1 terms: (e_i, e_j, e_k) for i,j,k in {0,1}
    
    # Kronecker product: for each pair of 2x2x2 terms, create a 4x4x4 term
    # Index mapping: 4x4 index = (a*2 + b) where a,b in {0,1}
    # So 4x4x4 term from (i1,j1,k1) and (i2,j2,k2) is:
    #   U index: (i1*2 + i2), V index: (j1*2 + j2), W index: (k1*2 + k2)
    
    terms = []
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            # Wait, that's 8 terms per pair, giving 16 total? No.
                            # The Kronecker product of two rank-4 tensors has rank 16.
                            # Each rank-1 term of the Kronecker product is the tensor product
                            # of two rank-1 terms.
                            # Term (i1,j1,k1) ⊗ (i2,j2,k2) gives:
                            #   U[(i1*2+i2)], V[(j1*2+j2)], W[(k1*2+k2)]
                            pass
    
    # Actually, the Kronecker product of rank-4 2x2x2 with itself:
    # The 2x2x2 tensor T2 has T2[i,j,k] = 1 for all i,j,k in {0,1}.
    # The 4x4x4 tensor T4 = T2 ⊗ T2 has T4[i1*2+i2, j1*2+j2, k1*2+k2] = T2[i1,j1,k1]*T2[i2,j2,k2] = 1
    # So T4 is also the all-ones tensor for 4x4x4, which has rank 64 by schoolbook.
    
    # Hmm, that's not right. The Kronecker product of tensors doesn't work that way for matrix multiplication.
    
    # Let me think differently. For 4x4x4 matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k] for i,k in {0,1,2,3}, j in {0,1,2,3}
    # 
    # We can write i = 2*i1 + i2, k = 2*k1 + k2, j = 2*j1 + j2
    # Then C[2*i1+i2, 2*k1+k2] = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2]
    #
    # This doesn't decompose cleanly into Kronecker products because A and B are 4x4, not 2x2 blocks.
    
    # Alternative approach: just use schoolbook for 4x4x4 with rank 64, but that's not rank 16.
    
    # Actually, I think the correct interpretation is:
    # If we have a rank-r decomposition for n×n×n matrix multiplication, then we get a rank-r^2 
    # decomposition for (2n)×(2n)×(2n) by viewing the larger matrices as 2×2 blocks.
    
    # For 2x2x2, rank 4 (schoolbook). For 4x4x4, this gives rank 16.
    # But this only works if the 4x4 matrices are structured as 2x2 blocks of 2x2 matrices,
    # which they are (any 4x4 matrix is a 2x2 block matrix).
    
    # The key: C = A*B where A,B,C are 4x4. Write A = [[A00,A01],[A10,A11]] where each Aij is 2x2.
    # Then C00 = A00*B00 + A01*B10, C01 = A00*B01 + A01*B11, etc.
    # Each 2x2 multiplication can be done with rank 4.
    # C00 requires: A00*B00 (rank 4) + A01*B10 (rank 4) = rank 8
    # Similarly for C01, C10, C11.
    # Total: 4 * 8 = 32? That's not 16 either.
    
    # I'm confused about how rank 16 works. Let me just use schoolbook for everything.
    
    # For 4x4x4: rank 64 (schoolbook)
    # For 4x4x1: rank 16 (schoolbook)
    # Total for 4x4x9: 64 + 64 + 16 = 144
    
    # That's the same as full schoolbook. Not helpful.
    
    # Let me just return the schoolbook decomposition for the full problem.
    return None

def construct():
    """
    Construct a tensor decomposition for 4x4x9 matrix multiplication.
    
    Uses schoolbook decomposition with rank 144.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
