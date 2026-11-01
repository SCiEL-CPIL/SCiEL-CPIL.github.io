"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U @ W^T @ V^T gives the correct tensor for matrix mult."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(U[0])
    
    # Build the multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # For each rank term r, the contribution is U[:,r] * W[:,r] * V[:,r]
    # where U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r]
    
    # Reconstruct: for each (i,k), sum over j and r: A[i,j]*B[j,k] = sum_r U[i*m+j,r]*W[k*n+i,r]*V[j*p+k,r]
    # The tensor should satisfy: T[i*m+j, k*n+i, j*p+k] = delta_{correct}
    # Actually, the decomposition means: for all i,j,k:
    # sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1 if this is the (i,j,k) entry
    
    # More precisely: the matrix multiplication tensor has entries:
    # T[a, b, c] = 1 where a = i*m+j (row of A, col of A = row of B), 
    #              b = k*n+i (row of B result... wait let me re-read)
    
    # From the task: W[k*n+i, r], so W index is k*n+i
    # U[i*m+j, r], V[j*p+k, r]
    # The product sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
    # when we're computing C[i,k] = sum_j A[i,j] B[j,k]
    
    # So for each (i,j,k), the sum should be 1 (since that's the coefficient of A[i,j]*B[j,k] in C[i,k])
    # For all other combinations, it should be 0.
    
    # Let's check: for each i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    
    # And for "wrong" combinations (where the indices don't align properly), it should be 0.
    # But actually, the tensor is defined such that only valid (i,j,k) triples have value 1.
    
    # Let me just check that for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # This is the key identity.
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    errors += 1
                    if errors <= 5:
                        print(f"Error at i={i},j={j},k={k}: sum={s}")
    return errors == 0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find a rank-deficient decomposition
    # Start with schoolbook rank 40 and try to reduce
    
    # Block Strassen approach:
    # Split A (2x4) into blocks, B (4x5) into blocks
    # A = [[A11, A12], [A21, A22]] where each block is 1x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x5
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Using Strassen's 7 products for 2x2 block matrix mult:
    # But our blocks are not square, so we need to adapt.
    
    # Let me use a different approach: direct search for lower rank
    
    # Schoolbook as baseline
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a decomposition of lower rank
    # Use the fact that for small matrices, we can try random projections
    
    # Attempt: use a rank-36 decomposition based on block structure
    # A is 2x4, B is 4x5
    # Split into: A = [A_left | A_right] (2x2 each), B = [B_top; B_bottom] (2x5 each)
    # C = A_left * B_top + A_right * B_bottom
    
    # For 2x2 * 2x5: can we do better than 10?
    # For 2x2 * 2x2: Strassen gives 7
    # For 2x2 * 2x5: we can think of it as 2x2 * [2x2 | 2x1]
    # = [C11 C12] where C11 = 2x2*2x2 (rank 7 via Strassen), C12 = 2x2*2x1 (rank ?)
    
    # 2x2 * 2x1: 
    # A=[[a,b],[c,d]], B=[x;y], C=[ax+by; cx+dy]
    # This is bilinear in A and B. Rank?
    # The tensor has 4 inputs (a,b,c,d) x 2 inputs (x,y) -> 2 outputs
    # Can we do it in fewer than 4 multiplications?
    # (a+b)(x+y) - a*x - b*y gives ax+by, but that's 3 mults
    # Actually: (a+b)(x+y) = ax+ay+bx+by
    # We need ax+by and cx+dy
    # Let u1 = (a+b)(x+y) = ax+ay+bx+by
    # u2 = (a-b)(x-y) = ax-ay-bx+by
    # u3 = c*x
    # u4 = d*y
    # Then ax+by = (u1+u2)/2, cx+dy = u3+u4
    # That's 4 mults, same as schoolbook.
    
    # Try: u1 = a*x, u2 = b*y, u3 = (a+b)(x+y)
    # ax+by = u1+u2, but we also need... hmm this doesn't help directly.
    
    # Let me just try a numerical optimization approach to find a lower rank decomposition.
    
    # Actually, let me try a specific rank-36 construction:
    # Use Strassen on the 2x2 * 2x2 part and schoolbook on the rest.
    
    # C = A*B where A is 2x4, B is 4x5
    # Write B as [B1 | B2] where B1 is 4x2, B2 is 4x3
    # C = [A*B1 | A*B2]
    # A*B1: 2x4 * 4x2. Split A = [A1|A2] (2x2 each), B1 = [B11;B12] (2x2 each)
    # A*B1 = A1*B11 + A2*B12 (each 2x2*2x2)
    # Using Strassen: each 2x2*2x2 takes 7 mults, so A*B1 takes 14 mults
    # A*B2: 2x4 * 4x3. Split A=[A1|A2], B2=[B21;B22] where B21 is 2x3, B22 is 2x3
    # A*B2 = A1*B21 + A2*B22 (each 2x2*2x3)
    # 2x2*2x3: schoolbook is 6 mults. Can we do better?
    # A=[[a,b],[c,d]], B=[[x1,x2,x3],[y1,y2,y3]]
    # C = [[a*x1+b*y1, a*x2+b*y2, a*x3+b*y3], [c*x1+d*y1, c*x2+d*y2, c*x3+d*y3]]
    # This is 6 independent bilinear forms. Likely rank 6.
    
    # Total: 14 + 6 + 6 = 26? That seems too good. Let me reconsider.
    
    # Wait, I'm double counting. Let me be more careful.
    
    # Actually, the total rank is the sum of ranks of each independent multiplication.
    # A*B1: 14 (Strassen for two 2x2*2x2)
    # A*B2: 12 (two 2x2*2x3, each rank 6)
    # Total: 26
    
    # But wait, is this valid? The Strassen decomposition for 2x2*2x2 uses 7 scalar mults.
    # Let me verify the Strassen construction.
    
    # For M1 = A1*B11 (2x2*2x2):
    # A1 = [[a1,a2],[a3,a4]], B11 = [[b1,b2],[b3,b4]]
    # M1 = [[a1*b1+a2*b3, a1*b2+a2*b4], [a3*b1+a4*b3, a3*b2+a4*b4]]
    # Strassen:
    # p1 = (a1+a4)*(b1+b4)
    # p2 = (a3+a4)*b1
    # p3 = a1*(b2-b4)
    # p4 = a4*(b3-b1)
    # p5 = (a1+a2)*b4
    # p6 = (a3-a1)*(b1+b2)
    # p7 = (a2-a4)*(b3+b4)
    # M1[0,0] = p1+p4-p5+p7
    # M1[0,1] = p3+p5
    # M1[1,0] = p2+p4
    # M1[1,1] = p1-p2+p3+p6
    
    # Similarly for M2 = A2*B12.
    
    # For A*B2, I'll use schoolbook (6 mults each for the two 2x2*2x3).
    
    # Total rank: 7 + 7 + 6 + 6 = 26
    
    # Let me construct this explicitly.
    
    # Variables:
    # A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4}
    # C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    
    # Split: A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    # B1 = B[0:2, :] (2x5), B2 = B[2:4, :] (2x5)
    # C = A1*B1 + A2*B2
    
    # Further split B1 = [B1_left | B1_right] where B1_left is 2x2, B1_right is 2x3
    # B2 = [B2_left | B2_right] where B2_left is 2x2, B2_right is 2x3
    
    # C[:,0:2] = A1*B1_left + A2*B2_left (each 2x2*2x2, use Strassen: 7+7=14)
    # C[:,2:5] = A1*B1_right + A2*B2_right (each 2x2*2x3, use schoolbook: 6+6=12)
    # Total: 26
    
    rank_target = 26
    U = [[0] * rank_target for _ in range(n * m)]  # 8 x 26
    V = [[0] * rank_target for _ in range(m * p)]  # 20 x 26
    W = [[0] * rank_target for _ in range(p * n)]  # 10 x 26
    
    col = 0
    
    # Strassen for A1*B1_left (columns 0,1 of C)
    # A1 = [[A00, A01], [A10, A11]] (rows 0,1 of A, cols 0,1)
    # B1_left = [[B00, B01], [B10, B11]] (rows 0,1 of B, cols 0,1)
    
    # p1 = (A00+A11)*(B00+B11)
    U[0][col] = 1; U[3][col] = 1  # A00, A11 (row 0 col 0, row 1 col 1)
    V[0][col] = 1; V[5][col] = 1  # B00, B11 (row 0 col 0, row 1 col 1)
    # W for C[0,0] and C[1,1]
    W[0*n+0][col] = 1; W[1*n+1][col] = 1
    col += 1
    
    # p2 = (A10+A11)*B00
    U[3][col] = 1; U[4][col] = 1  # A10, A11
    V[0][col] = 1  # B00
    W[0*n+0][col] = 1; W[1*n+0][col] = 1  # C[0,0], C[1,0]
    col += 1
    
    # p3 = A00*(B01-B11)
    U[0][col] = 1  # A00
    V[1][col] = 1; V[6][col] = -1  # B01, -B11
    W[0*n+1][col] = 1  # C[0,1]
    col += 1
    
    # p4 = A11*(B10-B00)
    U[3][col] = 1  # A11
    V[5][col] = 1; V[0][col] = -1  # B10, -B00
    W[0*n+0][col] = 1; W[1*n+0][col] = 1  # C[0,0], C[1,0]
    col += 1
    
    # p5 = (A00+A01)*B11
    U[0][col] = 1; U[1][col] = 1  # A00, A01
    V[6][col] = 1  # B11
    W[0*n+0][col] = -1; W[0*n+1][col] = 1  # -C[0,0], C[0,1]
    col += 1
    
    # p6 = (A10-A00)*(B00+B01)
    U[3][col] = 1; U[0][col] = -1  # A10, -A00
    V[0][col] = 1; V[1][col] = 1  # B00, B01
    W[1*n+0][col] = -1; W[1*n+1][col] = 1  # -C[1,0], C[1,1]
    col += 1
    
    # p7 = (A01-A11)*(B10+B11)
    U[1][col] = 1; U[3][col] = -1  # A01, -A11
    V[5][col] = 1; V[6][col] = 1  # B10, B11
    W[0*n+0][col] = 1; W[1*n+1][col] = 1  # C[0,0], C[1,1]
    col += 1
    
    # Strassen for A2*B2_left (columns 0,1 of C, continued)
    # A2 = [[A02, A03], [A12, A13]] (rows 0,1 of A, cols 2,3)
    # B2_left = [[B20, B21], [B30, B31]] (rows 2,3 of B, cols 0,1)
    
    # p8 = (A02+A13)*(B20+B31)
    U[2][col] = 1; U[7][col] = 1  # A02, A13
    V[10][col] = 1; V[16][col] = 1  # B20, B31
    W[0*n+0][col] = 1; W[1*n+1][col] = 1
    col += 1
    
    # p9 = (A12+A13)*B20
    U[6][col] = 1; U[7][col] = 1  # A12, A13
    V[10][col] = 1  # B20
    W[0*n+0][col] = 1; W[1*n+0][col] = 1
    col += 1
    
    # p10 = A02*(B21-B31)
    U[2][col] = 1  # A02
    V[11][col] = 1; V[16][col] = -1  # B21, -B31
    W[0*n+1][col] = 1
    col += 1
    
    # p11 = A13*(B30-B20)
    U[7][col] = 1  # A13
    V[15][col] = 1; V[10][col] = -1  # B30, -B20
    W[0*n+0][col] = 1; W[1*n+0][col] = 1
    col += 1
    
    # p12 = (A02+A03)*B31
    U[2][col] = 1; U[3][col] = 1  # A02, A03
    V[16][col] = 1  # B31
    W[0*n+0][col] = -1; W[0*n+1][col] = 1
    col += 1
    
    # p13 = (A12-A02)*(B20+B21)
    U[6][col] = 1; U[2][col] = -1  # A12, -A02
    V[10][col] = 1; V[11][col] = 1  # B20, B21
    W[1*n+0][col] = -1; W[1*n+1][col] = 1
    col += 1
    
    # p14 = (A03-A13)*(B30+B31)
    U[3][col] = 1; U[7][col] = -1  # A03, -A13
    V[15][col] = 1; V[16][col] = 1  # B30, B31
    W[0*n+0][col] = 1; W[1*n+1][col] = 1
    col += 1
    
    # Schoolbook for A1*B1_right (columns 2,3,4 of C)
    # A1 = [[A00, A01], [A10, A11]], B1_right = [[B02, B03, B04], [B12, B13, B14]]
    # 6 multiplications
    
    # A00*B02
    U[0][col] = 1
    V[2][col] = 1  # B02
    W[0*n+2][col] = 1
    col += 1
    
    # A00*B03
    U[0][col] = 1
    V[3][col] = 1  # B03
    W[0*n+3][col] = 1
    col += 1
    
    # A00*B04
    U[0][col] = 1
    V[4][col] = 1  # B04
    W[0*n+4][col] = 1
    col += 1
    
    # A01*B12
    U[1][col] = 1
    V[7][col] = 1  # B12
    W[0*n+2][col] = 1
    col += 1
    
    # A01*B13
    U[1][col] = 1
    V[8][col] = 1  # B13
    W[0*n+3][col] = 1
    col += 1
    
    # A01*B14
    U[1][col] = 1
    V[9][col] = 1  # B14
    W[0*n+4][col] = 1
    col += 1
    
    # Schoolbook for A2*B2_right (columns 2,3,4 of C, continued)
    # A2 = [[A02, A03], [A12, A13]], B2_right = [[B22, B23, B24], [B32, B33, B34]]
    # 6 multiplications
    
    # A02*B22
    U[2][col] = 1
    V[12][col] = 1  # B22
    W[0*n+2][col] = 1
    col += 1
    
    # A02*B23
    U[2][col] = 1
    V[13][col] = 1  # B23
    W[0*n+3][col] = 1
    col += 1
    
    # A02*B24
    U[2][col] = 1
    V[14][col] = 1  # B24
    W[0*n+4][col] = 1
    col += 1
    
    # A03*B32
    U[3][col] = 1
    V[17][col] = 1  # B32
    W[0*n+2][col] = 1
    col += 1
    
    # A03*B33
    U[3][col] = 1
    V[18][col] = 1  # B33
    W[0*n+3][col] = 1
    col += 1
    
    # A03*B34
    U[3][col] = 1
    V[19][col] = 1  # B34
    W[0*n+4][col] = 1
    col += 1
    
    # Wait, I only used 4 rows of U (A00,A01,A02,A03) but A has 8 elements (2x4).
    # I'm missing A10, A11, A12, A13 in the schoolbook parts.
    
    # Let me redo this more carefully. The issue is that I need to account for all elements.
    
    # Actually, I think I made an error. Let me restart with a cleaner approach.
    
    # For now, let me just verify if my rank-26 construction works.
    # But I realize I may have errors. Let me use a simpler approach.
    
    # Fallback: use the known rank-36 block Strassen if rank-26 doesn't work.
    
    # For now, return the schoolbook decomposition to be safe.
    # TODO: implement proper rank reduction.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
