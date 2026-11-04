"""Matrix multiplication 4x4x8 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 4x4x8 matrix multiplication tensor.
    
    Strategy: Use the fact that 4x4x8 multiplication can be viewed as
    two independent 4x4x4 multiplications, each decomposable via Strassen.
    A 4x4x4 tensor has rank 28 (two independent 2x2x2 Strassen instances).
    So 4x4x8 = 2 * 4x4x4 gives rank 56.
    
    But we can do better by exploiting the rectangular structure.
    We decompose the 4x4x8 tensor as a Kronecker product structure.
    
    Actually, let's use the direct approach:
    - The 4x4x8 tensor T[i,j,k] where i in [0,4), j in [0,4), k in [0,8)
    - We can split k into k = 2*k1 + k2 where k1 in [0,4), k2 in [0,2)
    - This gives us a 4x4x4x2 tensor
    
    Let me use a known result: the rank of the 4x4x8 tensor.
    Using Strassen's method generalized:
    - 2x2x2 has rank 7
    - 4x4x4 has rank 28 (two independent 2x2x2)
    - 4x4x8 can be seen as 4x4x4 x 2, which has rank 28 * 2 = 56 via naive
    - But with Strassen on the last dimension, we can get lower
    
    Let me construct it as:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Using the decomposition:
    - Split i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    - where i1,j1,k1 in {0,1}, i2,j2 in {0,1}, k2 in {0,1}
    
    The tensor becomes a 2x2x2 x 2x2x2 x 2x2x2 structure but with k having 4 elements.
    
    Actually, let me just use the known rank-56 construction first, then try to optimize.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use two independent 4x4x4 Strassen decompositions
    # Each 4x4x4 uses rank 28 (two 2x2x2 Strassen of rank 7)
    # Total: 2 * 28 = 56
    
    # First, define the 2x2x2 Strassen decomposition (rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    # For 2x2x2: A(2,2) * B(2,2) = C(2,2)
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Tensor: T[i,k,j] = 1
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M5 + M6
    
    # U[i,j, r] for A[i,j] coefficient in M_r
    # V[j,k, r] for B[j,k] coefficient in M_r  
    # W[k,i, r] for C[k,i] coefficient in M_r (note: W[k*n+i, r])
    
    # For 2x2x2: n=2, m=2, p=2
    # U has shape (n*m, 7) = (4, 7)
    # V has shape (m*p, 7) = (4, 7)
    # W has shape (p*n, 7) = (4, 7), W[k*n+i, r]
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 0],  # A00
        [0, 1, 0, 0, 0, 1, -1], # A01
        [0, 1, 0, 1, 1, -1, 0], # A10
        [1, 0, 0, 1, 0, 0, -1], # A11
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 0, 1],   # B01
        [1, 0, -1, 1, 1, 1, 0],  # B10
        [1, 0, 0, 0, 1, 0, 1],   # B11
    ]
    
    # W[k*n + i, r] where n=2
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M5 + M6
    strassen_W = [
        [1, 0, 0, -1, 0, 1, 1],   # C00: k=0, i=0
        [0, 1, 1, 0, 0, 0, 0],    # C01: k=0, i=1
        [0, 1, 0, 0, 1, 0, 0],    # C10: k=1, i=0
        [1, 0, -1, 0, 1, 1, 0],   # C11: k=1, i=1
    ]
    
    # Now for 4x4x4: split i=2*i1+i2, j=2*j1+j2, k=2*k1+k2
    # We have 4 independent 2x2x2 blocks: (i1,j1,k1) in {0,1}^3
    # But actually, it's 2 independent 2x2x2 Strassen instances for the 4x4x4
    # Wait, 4x4x4 = (2x2x2) ⊗ (2x2x2) in a sense
    # The rank of 4x4x4 is 28 = 7*4? No, 7*2=14? 
    # Actually, the rank of the 4x4x4 matrix multiplication tensor is 28.
    # This comes from: 4x4x4 can be decomposed as two independent 2x2x2 multiplications
    # for each half, giving 2 * 7 * 2 = 28.
    
    # Let me construct 4x4x4 with rank 28:
    # Split i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # For each (i1, j1, k1) combination, we have a 2x2x2 sub-tensor
    # But the structure is: T[i,j,k] = T[i1,j1,k1] * T[i2,j2,k2]
    # This is a Kronecker product, so rank = 7 * 7 = 49? No...
    
    # Actually, for 4x4x4:
    # T[i,j,k] where i,j,k in {0,1,2,3}
    # = T[i1,j1,k1] * T[i2,j2,k2] where i=2*i1+i2, etc.
    # The Kronecker product of two 2x2x2 tensors has rank 7*7=49
    # But the actual rank of 4x4x4 is 28, so there's a better decomposition.
    
    # The rank-28 decomposition of 4x4x4:
    # Use 4 independent Strassen decompositions, one for each (i1,j1) pair?
    # No, that would be 4*7=28. Let me think...
    
    # For 4x4x4: A(4,4) * B(4,4) = C(4,4)
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Ci,j is a 2x2 matrix = sum of two 2x2*2x2 products
    # We can use Strassen for each pair of products.
    # For C00 and C01: both involve A00, A01 and B00,B01,B10,B11
    # This is a 2x2x2x2 tensor problem...
    
    # Let me just use the naive rank-28 construction for 4x4x4:
    # 4 independent 2x2x2 Strassen decompositions
    # For (i1, j1) in {(0,0), (0,1), (1,0), (1,1)}:
    #   Compute A[i1*2:..., j1*2:...] * B[j1*2:..., k1*2:...] for k1 in {0,1}
    #   This gives 2 outputs per (i1,j1), so 4*2 = 8 sub-problems
    #   Each is 2x2x2 with rank 7
    #   Total: 8 * 7 / 2 = 28? 
    
    # I'm overcomplicating this. Let me just use the known result.
    # The rank of 4x4x4 is 28. Let me construct it directly.
    
    # For 4x4x8, the rank is at most 56 (two 4x4x4).
    # Let me construct rank-56 first, then see if I can do better.
    
    # 4x4x4 rank-28 construction:
    # I'll use the approach where we have 4 groups of 7 terms.
    # Group (i1, j1) handles A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k]
    
    # Actually, let me use a simpler approach:
    # For 4x4x8, use the schoolbook but with Strassen optimization on 2x2 blocks.
    # 
    # A(4,4) * B(4,8) = C(4,8)
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B into 2x8 blocks: B = [[B00, B01], [B10, B11]] where each is 2x8
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    #
    # Each Ci,j is 2x8 = sum of two 2x2 * 2x8 products
    # A 2x2 * 2x8 product can be done with rank... 
    # 2x2x8 tensor: A(2,2)*B(2,8)=C(2,8)
    # This has rank at most 2*2*8=32 naive
    # Using Strassen on the 2x2 part: rank 7 * 4 = 28? 
    # Actually, 2x2x8 = 2x2x2 x 4 (four independent 2x2x2)
    # So rank = 4 * 7 = 28
    #
    # For each Ci,j (4 of them), we need to compute two 2x2x8 products
    # But they share factors, so we can combine.
    #
    # Let me just go with rank 56 for now.
    
    # Construct 4x4x8 with rank 56:
    # Two independent 4x4x4 decompositions, each of rank 28.
    
    # 4x4x4 rank-28:
    # I'll construct it using 4 Strassen instances.
    # For i1 in {0,1}, j1 in {0,1}, k1 in {0,1}:
    #   Sub-problem: A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*2+k2] = C[i1*2+i2, k1*2+k2]
    #   This is a 2x2x2 problem with rank 7
    #   Total: 8 sub-problems * 7 = 56? No, that's too many.
    
    # Wait, I think I'm confusing myself. Let me be more careful.
    
    # For 4x4x4:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    # We want U[i,j,r], V[j,k,r], W[k,i,r] for r in {0,...,R-1}
    # such that T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # The rank of this tensor is 28.
    # One construction: use 4 independent 2x2x2 Strassen decompositions.
    # For (i1, j1) in {(0,0), (0,1), (1,0), (1,1)}:
    #   For k1 in {0,1}:
    #     Use Strassen to compute A[i1*2:..., j1*2:...] * B[j1*2:..., k1*2:...]
    #     This gives 2 output elements (for i2 in {0,1})
    #     Rank 7 per such sub-problem
    #   Total: 4 * 2 * 7 = 56? That's not 28.
    
    # I think the rank of 4x4x4 is actually 28, and the construction is:
    # 4 groups of 7, where group (i1, j1) handles all k values.
    # For fixed i1, j1:
    #   A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k] for all k
    #   This is a 2x4x4 problem (i2 in {0,1}, j2 in {0,1}, k in {0,1,2,3})
    #   The rank of 2x4x4 is 14? 
    #   4 * 14 = 56? Still not 28.
    
    # I'm clearly making an error. Let me just use the naive construction for now
    # and get a valid answer, then optimize.
    
    # Naive 4x4x8: rank 128
    # Let me at least get Strassen working for 2x2x2 and build up.
    
    # For now, let me return a valid rank-56 construction.
    # I'll use 8 independent 2x2x2 Strassen decompositions.
    
    U = [[0] * 56 for _ in range(16)]  # n*m = 4*4 = 16
    V = [[0] * 56 for _ in range(32)]  # m*p = 4*8 = 32
    W = [[0] * 56 for _ in range(32)]  # p*n = 8*4 = 32
    
    col = 0
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                # 2x2x2 Strassen sub-problem
                # A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*2+k2] = C[i1*2+i2, k1*2+k2]
                for r in range(7):
                    for i2 in range(2):
                        i = i1 * 2 + i2
                        U[i * 4 + (j1 * 2 + 0)][col] += strassen_U[i2 * 2 + 0][r]  # This is wrong
                        break
                # Let me redo this more carefully
                col += 7
    
    # This is getting complicated. Let me use a cleaner approach.
    # I'll construct the full decomposition directly.
    
    U = [[0] * 56 for _ in range(16)]
    V = [[0] * 56 for _ in range(32)]
    W = [[0] * 56 for _ in range(32)]
    
    col = 0
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                # For this 2x2x2 block, use Strassen
                for r in range(7):
                    # U: A[i, j] coefficient
                    for i2 in range(2):
                        i = i1 * 2 + i2
                        j = j1 * 2 + 0  # j2 = 0 for first, but we need both j2
                        # Actually, j is fixed for this sub-problem
                        # j = j1*2 + j2, but j2 varies
                        # Let me reconsider
                        pass
                    # This approach is flawed. Let me use a different strategy.
                    col += 1
    
    # Let me just use the schoolbook with a small optimization.
    # For 4x4x8, I'll use rank 128 (schoolbook) as a fallback.
    # But let me try to get at least rank 98 or so.
    
    # Actually, let me just return the schoolbook for now to have a valid answer.
    n, m, p = DIMENSIONS
    rank = n * m * p  # 128
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
# EVOLVE-BLOCK-END
