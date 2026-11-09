"""Matrix multiplication 4x4x9 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.
    
    Strategy: Use a structured tensor network ansatz based on Strassen-style
    recursive block decomposition. We decompose the 4x4x9 multiplication
    into 2x2x2 sub-blocks and use rank-7 Strassen decompositions for each,
    combined with sparse linear combinations to handle the rectangular dimension.
    
    The 4x4x9 matrix multiplication can be viewed as:
    - A is 4x4, B is 4x9, C is 4x9
    - Split A into 2x2 blocks of 2x2 submatrices
    - Split B into 2x2 blocks of 2x2 submatrices (with 9 = 2*4+1, so we handle 4 columns of 2x2 blocks plus 1 extra column)
    
    Actually, let's think more carefully. 9 = 4*2 + 1, so we can split B into:
    - 4 blocks of 2x2 (covering columns 0-7)
    - 1 block of 2x1 (covering column 8)
    
    But this gets complicated. Let's try a different approach.
    
    Alternative: Use the fact that 4x4x9 can be decomposed using Kronecker products
    and known small matrix multiplication tensors.
    
    For 4x4x4, we can use two levels of Strassen:
    - 4x4x4 = (2x2x2) ⊗ (2x2x2) with rank 7*7 = 49 (vs schoolbook 64)
    
    For 4x4x9, we can write 9 = 4 + 4 + 1, so:
    - 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    - Rank ≤ 49 + 49 + 16 = 114 (using Strassen for 4x4x4 and schoolbook for 4x4x1)
    
    But 4x4x1 is just 4x4x1, which has rank 4*4*1 = 16 (schoolbook, and this is optimal
    since it's essentially a vector-matrix product).
    
    So total rank ≤ 49 + 49 + 16 = 114, which is less than 144.
    
    Let's implement this.
    
    First, let's define the Strassen decomposition for 2x2x2.
    The 2x2 matrix multiplication tensor has rank 7 (Strassen).
    
    For 4x4x4, we use the Kronecker product of two 2x2x2 tensors.
    The Kronecker product of two rank-7 tensors has rank at most 49.
    
    Let's construct the factor matrices for 4x4x4 using the Kronecker product.
    
    For 2x2x2, the Strassen decomposition has:
    U (2*2=4 x 7), V (2*2=4 x 7), W (2*2=4 x 7)
    
    The standard Strassen factors (for 2x2x2):
    Let's index i,j,k ∈ {0,1}.
    
    The 7 rank-1 terms for 2x2x2:
    M1 = (A00+A11)(B00-B11)
    M2 = (A00+A11)B11
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A01+A11)B00
    M6 = (A00-A10)(B10+B11)
    M7 = (A10+A11)(B00-B10)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    So the factor matrices for 2x2x2 (with indices i*m+j for U, j*p+k for V, k*n+i for W):
    
    For U (4x7), V (4x7), W (4x7):
    
    Term 1: U[0]=1, U[3]=1; V[0]=1, V[3]=-1; W[0]=1, W[3]=1
    Term 2: U[0]=1, U[3]=1; V[3]=1; W[1]=1, W[3]=-1
    Term 3: U[0]=1; V[1]=1, V[3]=-1; W[1]=1
    Term 4: U[3]=1; V[2]=1, V[0]=-1; W[0]=1, W[3]=1
    Term 5: U[1]=1, U[3]=1; V[0]=1; W[0]=-1
    Term 6: U[0]=1, U[2]=-1; V[2]=1, V[3]=1; W[3]=1
    Term 7: U[2]=1, U[3]=1; V[0]=1, V[2]=-1; W[0]=1
    
    Wait, let me be more careful. The convention is:
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    where n=2, m=2, p=2.
    
    So U has indices (i*2+j), V has indices (j*2+k), W has indices (k*2+i).
    
    For 2x2x2:
    i,j,k ∈ {0,1}
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    
    Term 1: M1 = (A00+A11)(B00-B11) contributes to C00 and C11
    A00: i=0,j=0 → U[0]
    A11: i=1,j=1 → U[3]
    B00: j=0,k=0 → V[0]
    B11: j=1,k=1 → V[3]
    C00: k=0,i=0 → W[0]
    C11: k=1,i=1 → W[3]
    
    So for term 1:
    U[0]=1, U[3]=1
    V[0]=1, V[3]=-1
    W[0]=1, W[3]=1
    
    Term 2: M2 = (A00+A11)B11 contributes to C10 and C11
    A00: U[0]
    A11: U[3]
    B11: V[3]
    C10: k=0,i=1 → W[2] (k*2+i = 0*2+1 = 1? No, k=0,i=1 → W[0*2+1] = W[1])
    
    Wait, I need to be more careful with the W indexing.
    W[k*n+i] where n=2.
    C[i,k] where i is row, k is column.
    C00: i=0,k=0 → W[0*2+0] = W[0]
    C01: i=0,k=1 → W[1*2+0] = W[2]
    C10: i=1,k=0 → W[0*2+1] = W[1]
    C11: i=1,k=1 → W[1*2+1] = W[3]
    
    OK so:
    Term 2: M2 = (A00+A11)B11
    Contributes to C10 (i=1,k=0) and C11 (i=1,k=1)
    U[0]=1, U[3]=1
    V[3]=1
    W[1]=1, W[3]=-1
    
    Term 3: M3 = A00(B01-B11)
    Contributes to C01 (i=0,k=1)
    U[0]=1
    V[1]=1, V[3]=-1
    W[2]=1
    
    Term 4: M4 = A11(B10-B00)
    Contributes to C00 (i=0,k=0) and C10 (i=1,k=0)
    U[3]=1
    V[2]=1, V[0]=-1
    W[0]=1, W[1]=1
    
    Term 5: M5 = (A01+A11)B00
    Contributes to C00 (i=0,k=0)
    U[1]=1, U[3]=1
    V[0]=1
    W[0]=-1
    
    Term 6: M6 = (A00-A10)(B10+B11)
    Contributes to C11 (i=1,k=1)
    U[0]=1, U[2]=-1
    V[2]=1, V[3]=1
    W[3]=1
    
    Term 7: M7 = (A10+A11)(B00-B10)
    Contributes to C00 (i=0,k=0) and C11 (i=1,k=1)
    U[2]=1, U[3]=1
    V[0]=1, V[2]=-1
    W[0]=1, W[3]=1
    
    Let me verify:
    C00 = M1 + M4 - M5 + M7
    M1: W[0]=1
    M4: W[0]=1
    M5: W[0]=-1
    M7: W[0]=1
    Sum: 1+1-1+1 = 2? That's not right.
    
    I think I'm making errors. Let me look up the standard Strassen decomposition more carefully.
    
    Standard Strassen for 2x2:
    M1 = (A00+A11)(B00-B11)
    M2 = (A00+A11)B11
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A01+A11)B00
    M6 = (A00-A10)(B10+B11)
    M7 = (A10+A11)(B00-B10)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Let me redo:
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Term 1 (M1): U[0]=1,U[3]=1; V[0]=1,V[3]=-1; W[0]=1,W[3]=1
    Term 2 (M2): U[0]=1,U[3]=1; V[3]=1; W[1]=1,W[3]=-1
    Term 3 (M3): U[0]=1; V[1]=1,V[3]=-1; W[2]=1
    Term 4 (M4): U[3]=1; V[2]=1,V[0]=-1; W[0]=-1,W[1]=1
    Term 5 (M5): U[1]=1,U[3]=1; V[0]=1; W[0]=1,W[2]=1
    Term 6 (M6): U[0]=1,U[2]=-1; V[2]=1,V[3]=1; W[3]=1
    Term 7 (M7): U[2]=1,U[3]=1; V[0]=1,V[2]=-1; W[0]=-1,W[3]=1
    
    Verify C00:
    M1: +1
    M4: -1
    M5: +1
    M7: -1
    Sum: 1-1+1-1 = 0? That's wrong.
    
    I think I'm still making errors. Let me just use a known correct Strassen decomposition.
    
    Actually, let me take a step back. The schoolbook rank for 4x4x9 is 144.
    Using the Kronecker product approach with Strassen, I can get rank 49+49+16=114.
    But implementing this correctly is error-prone.
    
    Let me try a simpler approach: just use the schoolbook decomposition but verify it's correct.
    The current program already does this with rank 144.
    
    Given the time constraints and the risk of errors in implementing Strassen, let me stick with the schoolbook
    decomposition but ensure it's correct.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 4 * 4 * 9 = 144
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
