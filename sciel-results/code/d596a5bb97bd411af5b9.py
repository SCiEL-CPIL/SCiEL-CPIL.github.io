import numpy as np
import time

def construct():
    t0 = time.time()
    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 base tensors (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # c=0
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # c=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    # c=2
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    # c=3
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    # c=4
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    # c=5
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # c=6
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    # Precompute the 4x4*4x4 Strassen decomposition (rank 49)
    # For a 4x4 matrix A * 4x4 matrix B = C
    # We split into 2x2 blocks and apply Strassen to each
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 4x4*4x4 decomposition has rank 49
    # We build it by applying Strassen at the 2x2 level
    # Each 2x2 block multiplication uses rank 7
    
    # For 4x4*4x4:
    # C00 = A00*B00 + A01*B10  (two 2x2*2x2 multiplications)
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # But we can use Strassen on the 2x2 blocks:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    # C00 = M0+M3-M5+M6
    # C01 = M2+M4
    # C10 = M1+M5
    # C11 = M0-M1+M2+M4
    
    # For 4x4*4x8, we split B into two 4x4 blocks:
    # B = [B_left, B_right] where B_left is 4x4 and B_right is 4x4
    # C = A * B = [A*B_left, A*B_right]
    
    # So we do two independent 4x4*4x4 multiplications, each of rank 49
    # Total rank = 98
    
    # Build the 4x4*4x4 decomposition first
    # Indexing: A has 16 entries (4x4), B has 16 entries (4x4), C has 16 entries (4x4)
    
    # For the 4x4*4x4 case:
    # A is indexed as A[i,j] for i,j in 0..3
    # B is indexed as B[j,k] for j,k in 0..3
    # C is indexed as C[i,k] for i,k in 0..3
    
    # Split into 2x2 blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # Each 2x2 block multiplication uses the 7-term Strassen formula
    # So for each of the 7 Strassen terms, we need to build the U, V, W contributions
    
    # Let's precompute the 4x4*4x4 decomposition
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # For each Strassen term s in 0..6:
    # The term involves specific 2x2 blocks of A, B, C
    # We need to map the 2x2 block indices to the full 4x4 indices
    
    # Strassen terms:
    # s=0: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    # s=1: (A10+A11)*B00 -> C10 += 1, C11 -= 1
    # s=2: A00*(B01-B11) -> C01 += 1, C11 += 1
    # s=3: A11*(B10-B00) -> C00 += 1, C10 += 1
    # s=4: (A00+A01)*B11 -> C01 += 1, C11 -= 1
    # s=5: (A10-A00)*(B00+B01) -> C00 -= 1, C10 += 1
    # s=6: (A01-A11)*(B10+B11) -> C00 += 1, C11 += 1
    
    # For each term, the U, V, W matrices are built from the 2x2 Strassen base
    # The 2x2 Strassen base has U2, V2, W2 as defined above
    
    # Let's define the block mappings:
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B blocks: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C blocks: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    
    # For each Strassen term, specify which A blocks, B blocks, C blocks are involved
    # and with what coefficients
    
    strassen_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # Each block is (coeff, block_row, block_col)
        ([(1,0,0), (1,1,1)], [(1,0,0), (1,1,1)], [(1,0,0), (1,1,1)]),  # s=0
        ([(1,1,0), (1,1,1)], [(1,0,0)], [(1,1,0), (-1,1,1)]),  # s=1
        ([(1,0,0)], [(1,0,1), (-1,1,1)], [(1,0,1), (1,1,1)]),  # s=2
        ([(1,1,1)], [(1,1,0), (-1,0,0)], [(1,0,0), (1,1,0)]),  # s=3
        ([(1,0,0), (1,0,1)], [(1,1,1)], [(1,0,1), (-1,1,1)]),  # s=4
        ([(1,1,0), (-1,0,0)], [(1,0,0), (1,0,1)], [(-1,0,0), (1,1,0)]),  # s=5
        ([(1,0,1), (-1,1,1)], [(1,1,0), (1,1,1)], [(1,0,0), (1,1,1)]),  # s=6
    ]
    
    # Build the 4x4*4x4 decomposition
    for s in range(7):
        a_blocks, b_blocks, c_blocks = strassen_terms[s]
        for k in range(7):  # k indexes the 2x2 Strassen sub-terms
            col4 = s * 7 + k
            for coeff_a, ar, ac in a_blocks:
                for r in range(2):
                    for c in range(2):
                        gi = ar * 2 + r
                        gj = ac * 2 + c
                        a_idx = gi * 4 + gj
                        a_eff = r * 2 + c
                        U4[a_idx, col4] += coeff_a * U2[a_eff, k]
            for coeff_b, br, bc in b_blocks:
                for r in range(2):
                    for c in range(2):
                        gj = br * 2 + r
                        gk = bc * 2 + c
                        b_idx = gj * 4 + gk
                        b_eff = r * 2 + c
                        V4[b_idx, col4] += coeff_b * V2[b_eff, k]
            for coeff_c, cr, cc in c_blocks:
                for r in range(2):
                    for c in range(2):
                        gi = cr * 2 + r
                        gk = cc * 2 + c
                        c_idx = gk * 4 + gi
                        c_eff = r * 2 + c
                        W4[c_idx, col4] += coeff_c * W2[c_eff, k]
    
    # Now build the 4x4*4x8 decomposition
    # A is 4x4, B is 4x8, C is 4x8
    # Split B into B_left (4x4) and B_right (4x4)
    # C = [C_left, C_right] where C_left = A*B_left, C_right = A*B_right
    
    # For the left half (columns 0-3 of B and C):
    # Use the 4x4*4x4 decomposition directly
    # A indices: 0..15 (row-major 4x4)
    # B_left indices: 0..15 (row-major 4x4, columns 0-3 of B)
    # C_left indices: 0..15 (row-major 4x4, columns 0-3 of C)
    
    # For the right half (columns 4-7 of B and C):
    # Use the 4x4*4x4 decomposition
    # A indices: 0..15 (same A)
    # B_right indices: 16..31 (row-major 4x4, columns 4-7 of B)
    # C_right indices: 16..31 (row-major 4x4, columns 4-7 of C)
    
    # Fill U, V, W for the full 4x4*4x8 decomposition
    # U has shape (16, 98)
    # V has shape (32, 98)
    # W has shape (32, 98)
    
    # Left half: columns 0-47 of U, V, W
    # A part: same for both halves
    U[:, 0:49] = U4
    # B part: columns 0-15 of B
    V[0:16, 0:49] = V4
    # C part: columns 0-15 of C (W is indexed as C[k*n+i,r] = C[k*4+i])
    # Wait, the task says W[k*n+i,r] where n=4, so W is indexed as W[k*4+i, r]
    # For C_left, k=0..3, i=0..3, so W[0:16, 0:49] = W4
    W[0:16, 0:49] = W4
    
    # Right half: columns 48-97 of U, V, W
    # A part: same A
    U[:, 48:98] = U4
    # B part: columns 16-31 of B
    V[16:32, 48:98] = V4
    # C part: columns 16-31 of C
    W[16:32, 48:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
