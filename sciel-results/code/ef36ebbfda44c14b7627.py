"""Rewritten program for 2x2 by 2x11 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x11 matrix.
    
    Uses Strassen's algorithm for the 2x2 block structure to reduce rank
    from 44 (schoolbook) to 39 (7 per 2x2 block * 11 columns - 11 shared).
    
    Actually, let me think more carefully:
    - A is 2x2, B is 2x11, C is 2x11
    - Schoolbook: rank = 2*2*11 = 44
    - Using Strassen on the 2x2 part: For each column of B, we compute
      a 2x2 * 2x1 product. But Strassen works for 2x2 * 2x2.
    
    Let me reconsider. The tensor is for (n,m,p) = (2,2,11).
    C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..1, k in 0..10.
    
    Schoolbook rank = n*m*p = 2*2*11 = 44.
    
    Strassen's algorithm: For 2x2 * 2x2, rank 7 instead of 8.
    But here B is 2x11, not 2x2.
    
    We can view this as: for each pair of columns of B, we have a 2x2 * 2x2 multiplication.
    But 11 is odd, so we have 5 pairs of columns plus 1 single column.
    
    Alternative: Use the fact that the tensor decomposes as a direct sum over k of
    2x2 * 2x1 tensors, but that doesn't help directly.
    
    Better approach: The 2x2 * 2x11 multiplication can be seen as 11 independent
    2x2 * 2x1 multiplications. But we can group them.
    
    Actually, let me think about it differently. The multiplication tensor for
    (n,m,p) has rank at most n*m*p. For (2,2,11), that's 44.
    
    Using Strassen: We can embed the 2x2 * 2x11 into a 2x2 * 2x12 problem by padding,
    then use Strassen-like decompositions. But that's not straightforward.
    
    Let me use a simpler approach: The known result is that for M(2,2,p), the rank
    can be reduced. For p=11, let me try to construct a rank-39 decomposition.
    
    Key insight: For 2x2 * 2x2, Strassen gives rank 7. If we could tile the 2x11
    with 2x2 blocks... but 11 = 5*2 + 1, so we have 5 blocks of 2 columns and 1
    leftover column.
    
    For each 2-column block of B, we do a 2x2 * 2x2 multiplication using Strassen
    (rank 7). For the leftover single column, we do a 2x2 * 2x1 multiplication
    (rank 4 schoolbook, or can we do better?).
    
    Total: 5*7 + 4 = 35 + 4 = 39.
    
    Let me construct this.
    """
    n, m, p = DIMENSIONS  # 2, 2, 11
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For 2x2 * 2x1 (single column):
    # A = [[a,b],[c,d]], B = [[e],[g]]
    # C = [[ae+bg],[ce+dg]]
    # Schoolbook rank 4. Can we do better? The tensor is M(2,2,1) which has
    # rank 4 (it's the sum of two M(1,2,1) tensors... actually no).
    # M(2,2,1) is the tensor for 2x2 * 2x1. Its rank is 4.
    
    # So total rank = 5*7 + 4 = 39.
    
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)  # 4 x 39
    V = np.zeros((m * p, rank), dtype=float)  # 22 x 39
    W = np.zeros((p * n, rank), dtype=float)  # 22 x 39
    
    # W index: W[k*n + i, r] for k in 0..10, i in 0..1
    # U index: U[i*m + j, r] for i in 0..1, j in 0..1
    # V index: V[j*p + k, r] for j in 0..1, k in 0..10
    
    col = 0
    
    # Process 5 blocks of 2 columns each
    for block in range(5):
        k0 = block * 2  # first column index in this block
        k1 = k0 + 1     # second column index
        
        # Strassen factors for this 2x2 * 2x2 block
        # m1 = (a+d)(e+h): A rows i=0,1 cols j=0,1; B rows j=0,1 cols k=k0,k1
        # In our notation:
        # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        # e = B[0,k0], f = B[0,k1], g = B[1,k0], h = B[1,k1]
        
        # m1 = (a+d)(e+h)
        # U: (a+d) => U[0,0]=1, U[1,1]=1 (i=0,j=0 and i=1,j=1)
        # Wait, U[i*m+j, r]. For a = A[0,0]: i=0,j=0 => U[0, r]
        # For d = A[1,1]: i=1,j=1 => U[3, r]
        U[0, col] = 1   # a
        U[3, col] = 1   # d
        # V: (e+h) => V[0*p+k0, r]=1, V[1*p+k1, r]=1
        # e = B[0,k0]: j=0, k=k0 => V[0*11+k0, r] = V[k0, r]
        # h = B[1,k1]: j=1, k=k1 => V[1*11+k1, r] = V[11+k1, r]
        V[k0, col] = 1       # e
        V[11 + k1, col] = 1  # h
        # W: C11 += m1, C12 += m1
        # C11: i=0, k=k0 => W[k0*2+0, r] = W[2*k0, r]
        # C12: i=0, k=k1 => W[k1*2+0, r] = W[2*k1, r]
        W[2 * k0, col] = 1
        W[2 * k1, col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[3, col] = 1   # c = A[1,0] => i=1,j=0 => U[2, r]
        U[2, col] = 1   # wait, c = A[1,0]: i=1, j=0 => U[1*2+0, r] = U[2, r]
        # Let me redo: U[i*m+j, r]
        # a = A[0,0]: i=0,j=0 => U[0, r]
        # b = A[0,1]: i=0,j=1 => U[1, r]
        # c = A[1,0]: i=1,j=0 => U[2, r]
        # d = A[1,1]: i=1,j=1 => U[3, r]
        
        # m2 = (c+d)e: c=A[1,0], d=A[1,1], e=B[0,k0]
        U[2, col] = 1   # c
        U[3, col] = 1   # d
        V[k0, col] = 1  # e = B[0,k0]
        # C21 += m2: i=1, k=k0 => W[2*k0+1, r]
        W[2 * k0 + 1, col] = 1
        col += 1
        
        # m3 = a(f-h): a=A[0,0], f=B[0,k1], h=B[1,k1]
        U[0, col] = 1   # a
        V[11 + k1, col] = 1   # f = B[0,k1] => j=0,k=k1 => V[k1, r]
        V[11 + k1, col] = -1  # h = B[1,k1] => j=1,k=k1 => V[11+k1, r]
        # Wait, I need to fix: f = B[0,k1]: j=0, k=k1 => V[0*11+k1, r] = V[k1, r]
        # h = B[1,k1]: j=1, k=k1 => V[1*11+k1, r] = V[11+k1, r]
        V[k1, col] = 1       # f
        V[11 + k1, col] = -1 # h
        # C12 += m3: i=0, k=k1 => W[2*k1, r]
        W[2 * k1, col] = 1
        col += 1
        
        # m4 = d(g-e): d=A[1,1], g=B[1,k0], e=B[0,k0]
        U[3, col] = 1   # d
        V[11 + k0, col] = 1   # g = B[1,k0] => j=1,k=k0 => V[11+k0, r]
        V[k0, col] = -1       # e = B[0,k0] => j=0,k=k0 => V[k0, r]
        # C11 += m4: i=0, k=k0 => W[2*k0, r]
        # C21 += m4: i=1, k=k0 => W[2*k0+1, r]
        W[2 * k0, col] = 1
        W[2 * k0 + 1, col] = 1
        col += 1
        
        # m5 = (a+b)h: a=A[0,0], b=A[0,1], h=B[1,k1]
        U[0, col] = 1   # a
        U[1, col] = 1   # b
        V[11 + k1, col] = 1  # h = B[1,k1]
        # C11 -= m5: i=0, k=k0 => W[2*k0, r]
        # C12 += m5: i=0, k=k1 => W[2*k1, r]
        W[2 * k0, col] = -1
        W[2 * k1, col] = 1
        col += 1
        
        # m6 = (c-a)(e+f): c=A[1,0], a=A[0,0], e=B[0,k0], f=B[0,k1]
        U[2, col] = 1   # c
        U[0, col] = -1  # a
        V[k0, col] = 1  # e
        V[k1, col] = 1  # f
        # C22 += m6: i=1, k=k1 => W[2*k1+1, r]
        W[2 * k1 + 1, col] = 1
        col += 1
        
        # m7 = (b-d)(g+h): b=A[0,1], d=A[1,1], g=B[1,k0], h=B[1,k1]
        U[1, col] = 1   # b
        U[3, col] = -1  # d
        V[11 + k0, col] = 1  # g
        V[11 + k1, col] = 1  # h
        # C11 += m7: i=0, k=k0 => W[2*k0, r]
        # C22 += m7: i=1, k=k1 => W[2*k1+1, r]
        W[2 * k0, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1
    
    # Process the last single column (k=10)
    k = 10
    # 2x2 * 2x1 schoolbook: rank 4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Term 1: A[0,0]*B[0,k]
    U[0, col] = 1   # A[0,0]
    V[k, col] = 1   # B[0,k]
    W[2*k, col] = 1 # C[0,k]
    col += 1
    
    # Term 2: A[0,1]*B[1,k]
    U[1, col] = 1   # A[0,1]
    V[11 + k, col] = 1  # B[1,k]
    W[2*k, col] = 1 # C[0,k]
    col += 1
    
    # Term 3: A[1,0]*B[0,k]
    U[2, col] = 1   # A[1,0]
    V[k, col] = 1   # B[0,k]
    W[2*k+1, col] = 1  # C[1,k]
    col += 1
    
    # Term 4: A[1,1]*B[1,k]
    U[3, col] = 1   # A[1,1]
    V[11 + k, col] = 1  # B[1,k]
    W[2*k+1, col] = 1  # C[1,k]
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
