import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    # Strategy: Split into overlapping 2x2 blocks of B
    # For each adjacent pair of columns in A, use Strassen on 2x2 subblocks of B
    # 
    # A has columns 0,1,2. We can form pairs (0,1) and (1,2).
    # For pair (0,1): A[:,0:2] is 2x2, multiply with B[0:2, :] which is 2x10
    # For pair (1,2): A[:,1:3] is 2x2, multiply with B[1:3, :] which is 2x10
    #
    # But this double-counts column 1. Let's be more careful.
    #
    # Alternative: Use the fact that C = A * B where A is 2x3, B is 3x10.
    # Write A = [A0, A1, A2] where Ai is 2x1.
    # C = A0 * B0 + A1 * B1 + A2 * B2 where Bk is row k of B (1x10)
    #
    # Each Ai * Bk is an outer product of 2x1 and 1x10, which has rank 1.
    # So total rank is 6, but that's just summing 6 rank-1 terms.
    # Wait, that's not right. The tensor decomposition requires that each "rank-1" term
    # corresponds to a single multiplication.
    #
    # Actually, for matrix multiplication, the minimal rank over the reals is known.
    # For 2x3x10, let's think about it differently.
    #
    # Let me use a direct approach: exploit structure by grouping.
    # Group B's columns into pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each pair of columns, we have a 3x2 submatrix of B.
    # A * B_pair gives a 2x2 output.
    #
    # So we have 5 blocks, each computing a 2x2 product from 2x3 * 3x2.
    # The 2x2x4 problem (2x2 times 2x2 is 2x2, but here it's 2x3 times 3x2)
    #
    # For 2x3 times 3x2:
    # Let A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C = [[ag+bi+ck, ah+bj+cl], [dg+ei+fk, dh+ej+fl]]
    #
    # We can use Strassen-like: 
    # M1 = (a+b)(g+h) -> but dimensions don't match nicely
    #
    # Let's just use the standard approach with sharing.
    # For each 3x2 block of B (5 blocks), compute 2x2 product with A.
    # Each 2x3 x 3x2 product: what's the minimal rank?
    #
    # Actually, let me just implement a valid decomposition first, then optimize.
    # The trivial one: for each (i,j,k), U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # That's rank 60.
    #
    # Let me try to reduce by noting that for fixed j, the i and k indices are independent.
    # For fixed j, we have a 2x10 outer product of A[:,j] and B[j,:].
    # This can be done with rank 20 (each element is a separate product).
    # But we can share: A[:,j] is 2x1, B[j,:] is 1x10.
    # The outer product has rank 1 as a matrix, but as a tensor decomposition element...
    #
    # I think the key insight is: we can use bilinear forms.
    # Let me define linear combinations of A's entries and B's entries, multiply them,
    # and combine to get C's entries.
    #
    # For a 2x3x10 problem, let me try rank 50 as a target.
    #
    # Simpler approach: Use the fact that 2x3x10 = 2x2x10 + 2x1x10
    # where we split A into first 2 columns and last 1 column.
    # 2x2x10: For each pair of rows in B (5 pairs of 2 columns), do 2x2x2 with Strassen (rank 7)
    #   So 5 * 7 = 35
    # 2x1x10: outer product of 2x1 and 1x10, rank 20
    # Total: 55
    #
    # But we also need to handle the middle column shared between the two 2x2 blocks? No, 
    # if we split A as [A0:2, A2] where A0:2 is first 2 cols, A2 is last col.
    # Then B0:2 is first 2 rows of B (3x2), B2 is last row (1x10).
    # Wait, B is 3x10, so rows are the middle dimension.
    #
    # A is 2x3: rows i=0,1; cols j=0,1,2
    # B is 3x10: rows j=0,1,2; cols k=0,...,9
    # C is 2x10: rows i=0,1; cols k=0,...,9
    #
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # Split into j=0,1 and j=2:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + A[i,2] * B[2,k]
    #
    # First term: 2x2x10 (A[:,0:2] is 2x2, B[0:2,:] is 2x10)
    # Second term: 2x1x10 (A[:,2] is 2x1, B[2,:] is 1x10)
    #
    # For 2x2x10: Split B's columns into 5 pairs of 2. For each pair, do 2x2x2.
    # 2x2x2 has minimal rank 7 (Strassen). So 5*7 = 35.
    # For 2x1x10: outer product, each C[i,k] = A[i,2]*B[2,k], so rank 20 (one per output element).
    # Total: 35 + 20 = 55.
    #
    # Can we do better? Maybe overlap the j=1 column.
    # Use A[:,0:2] for j=0,1 and A[:,1:3] for j=1,2, then subtract the double count.
    # But that gets complicated.
    #
    # Let's go with 55 for now.
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)  # (6, 55)
    V = np.zeros((m * p, rank), dtype=float)  # (30, 55)
    W = np.zeros((p * n, rank), dtype=float)  # (20, 55)
    
    col = 0
    # Part 1: 2x2x10 using Strassen on 5 blocks of 2x2x2
    # A_sub = A[:, 0:2] (2x2), B_sub blocks = B[0:2, 2l:2l+2] (2x2)
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Strassen's 7 products for 2x2x2:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # U indices: i*m + j for i in {0,1}, j in {0,1} -> rows 0,1,3,4
        # V indices: j*p + k for j in {0,1}, k in {k0,k1} -> rows 0,1,p,k0+p,k1+p
        # W indices: k*n + i for k in {k0,k1}, i in {0,1}
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0, col] = 1  # A[0,0]
        U[4, col] = 1  # A[1,1]
        V[0 + k0, col] = 1  # B[0,k0]
        V[1 + k1, col] = 1  # B[1,k1]
        # C contributions: C00 += M1, C11 += M1
        W[k0 * n + 0, col] = 1  # C[0,k0]
        W[k1 * n + 1, col] = 1  # C[1,k1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3, col] = 1  # A[1,0]
        U[4, col] = 1  # A[1,1]
        V[0 + k0, col] = 1  # B[0,k0]
        # C10 += M2
        W[k0 * n + 1, col] = 1  # C[1,k0]
        col += 1
        
        # M3 = A[0,0] * (B[1,k0]-B[1,k1])
        U[0, col] = 1  # A[0,0]
        V[1 + k0, col] = 1  # B[1,k0]
        V[1 + k1, col] = -1  # -B[1,k1]
        # C01 += M3
        W[k1 * n + 0, col] = 1  # C[0,k1]
        col += 1
        
        # M4 = A[1,1] * (B[0,k1]-B[0,k0])
        U[4, col] = 1  # A[1,1]
        V[0 + k1, col] = 1  # B[0,k1]
        V[0 + k0, col] = -1  # -B[0,k0]
        # C00 += M4, C10 += M4
        W[k0 * n + 0, col] = 1  # C[0,k0]
        W[k0 * n + 1, col] = 1  # C[1,k0]
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0, col] = 1  # A[0,0]
        U[1, col] = 1  # A[0,1]
        V[1 + k1, col] = 1  # B[1,k1]
        # C00 -= M5, C01 += M5
        W[k0 * n + 0, col] = -1  # C[0,k0]
        W[k1 * n + 0, col] = 1  # C[0,k1]
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[1,k0])
        U[3, col] = 1  # A[1,0]
        U[0, col] = -1  # -A[0,0]
        V[0 + k0, col] = 1  # B[0,k0]
        V[1 + k0, col] = 1  # B[1,k0]
        # C11 += M6
        W[k1 * n + 1, col] = 1  # C[1,k1]
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[0,k1]+B[1,k1])
        U[1, col] = 1  # A[0,1]
        U[4, col] = -1  # -A[1,1]
        V[0 + k1, col] = 1  # B[0,k1]
        V[1 + k1, col] = 1  # B[1,k1]
        # C00 += M7
        W[k0 * n + 0, col] = 1  # C[0,k0]
        col += 1
    
    # Part 2: 2x1x10 for the third column of A and third row of B
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(10):
            U[i * m + 2, col] = 1  # A[i,2], row = i*3+2
            V[2 * p + k, col] = 1  # B[2,k], row = 2*10+k
            W[k * n + i, col] = 1  # C[i,k]
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

