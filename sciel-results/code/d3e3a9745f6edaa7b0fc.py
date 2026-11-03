import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    rank = 37
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 1: 2x4x4 using rank-29 (block Strassen)
    # A = [[a00,a01,a02,a03],[a10,a11,a12,a13]]
    # B_left = first 4 cols of B (4x4)
    # C_left = A * B_left (2x4)
    #
    # Split: A_L = cols 0-1, A_R = cols 2-3
    # B_LL = B[0:2,0:2], B_LR = B[0:2,2:4], B_RL = B[2:4,0:2], B_RR = B[2:4,2:4]
    # C[0,0:2] = A_L*B_LL + A_R*B_RL
    # C[0,2:4] = A_L*B_LR + A_R*B_RR
    # C[1,0:2] = A_L*B_LL + A_R*B_RL (same as row 0 but with A row 1)
    # Actually C[i,0:2] = A[i,0:2]*B_LL + A[i,2:4]*B_RL
    # C[i,2:4] = A[i,0:2]*B_LR + A[i,2:4]*B_RR
    
    # Strassen for 2x2*2x2: given P = X*Y where X,Y are 2x2
    # S1=(x00+x11)(y00+y11), S2=(x10+x11)y00, S3=x00(y10-y11),
    # S4=x11(y01-y00), S5=(x00+x01)y11, S6=(x10-x00)(y00+y10), S7=(x01-x11)(y01+y11)
    # P00=S1+S4-S5+S7, P01=S3+S5, P10=S2+S4, P11=S1-S2+S3+S6
    
    # For C[i,0:2] = A[i,0:2]*B_LL + A[i,2:4]*B_RL
    # We need to compute two 2x2 products and add.
    # Using Strassen on each: 7+7=14 terms per row of C, but A is shared.
    # 
    # Let me define 14 Strassen terms for row i of C_left:
    # For product A[i,0:2] * B_LL:
    #   t1 = (A[i,0]+A[i,1])*(B_LL[0,0]+B_LL[1,1])
    #   t2 = (A[i,1]+A[i,1])*B_LL[0,0]  -- wait, this doesn't work for 1x4 * 4x2
    
    # I'm making this too complicated. Let me use a simpler approach:
    # Direct schoolbook for 2x4x4 gives 2*4*4=32 terms.
    # Then 2x4x2 schoolbook gives 8 terms. Total 40.
    # That's not better than 42.
    
    # Let me try: schoolbook for 2x4x6 directly = 2*4*6 = 48. No.
    
    # Alternative: Use Strassen on 2x2x2 blocks.
    # Split A into 2 blocks of 2x4: A_top (1x4), A_bot (1x4)
    # Split B into 3 blocks of 4x2: B1 (cols 0-1), B2 (cols 2-3), B3 (cols 4-5)
    # C = [A_top*B1 + A_bot*B1, ...] -- no, this is wrong.
    
    # Correct: C[i,k] = sum_j A[i,j]*B[j,k]
    # Split j into {0,1} and {2,3}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #        = (A[i,0],A[i,1]) * B[0:2,k] + (A[i,2],A[i,3]) * B[2:4,k]
    
    # For each i and k, this is a dot product of two 2-vectors plus two 2-vectors.
    # Using Strassen-like: for 1x2 * 2x1, we need 3 terms (not 2) to save?
    # No, 1x2*2x1 is just a dot product, needs 2 terms minimum.
    
    # Let me try a different split. Group columns of B in pairs:
    # For output column k and k+1 (k even):
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i,k+1] = sum_j A[i,j]*B[j,k+1]
    # 
    # This is A[i,:] * B[:,k:k+2], a 1x4 * 4x2 = 1x2 result.
    # Schoolbook: 4*2 = 8 terms per (i, k_pair).
    # Total: 2 rows * 3 col_pairs * 8 = 48. Worse.
    
    # I think the best I can do with simple methods is:
    # - 2x4x4 at rank 29 + 2x4x2 at rank 8 = 37
    # Let me construct the rank-29 for 2x4x4 properly.
    
    # Known: 2x4x4 has rank at most 29 (Bini et al.)
    # Construction: View as 2*(2x2x4) or use block methods.
    
    # Simplest correct approach: I'll use 2x4x4 at rank 32 (schoolbook) 
    # and 2x4x2 at rank 8, total 40. Still better than 42.
    
    # Actually wait - let me just do full schoolbook 2x4x6 = 48. No.
    
    # Let me reconsider. The current program has rank 42.
    # Schoolbook 2*4*6 = 48.
    # I need to beat 42.
    
    # Strategy: 2x4x4 at rank 28 + 2x4x2 at rank 8 = 36? 
    # Or 2x4x4 at rank 29 + 2x4x2 at rank 8 = 37.
    
    # For 2x4x4, let me use: split into two 2x2x4 problems.
    # 2x2x4: A is 2x2, B is 2x4. C = A*B is 2x4.
    # Schoolbook: 2*2*4 = 16.
    # Can we do better? Split B into two 2x2 blocks: B = [B1 B2]
    # C = [A*B1, A*B2]. Each 2x2x2 at rank 7. Total 14.
    # So 2x4x4 = two 2x2x4 = 2*14 = 28.
    # Plus 2x4x2 = schoolbook 8.
    # Total: 28 + 8 = 36!
    
    # Let me implement this:
    # A is 2x4, split into A_top (row 0) and A_bot (row 1).
    # Wait, A is 2x4, so A_top is 1x4, A_bot is 1x4.
    # 
    # C = A * B where B is 4x6.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Split B into B_left (4x4) and B_right (4x2).
    # C_left = A * B_left (2x4), C_right = A * B_right (2x2).
    # 
    # For C_left: split A into A[0,:] (1x4) and A[1,:] (1x4).
    # C_left[0,k] = A[0,:] * B_left[:,k]
    # C_left[1,k] = A[1,:] * B_left[:,k]
    # 
    # This doesn't directly give 2x2x4 structure.
    
    # Alternative: Split B_left into top 2 rows and bottom 2 rows.
    # B_left = [B_top; B_bot] where B_top is 2x4, B_bot is 2x4.
    # C_left[i,k] = A[i,0]*B_top[0,k] + A[i,1]*B_top[1,k] + A[i,2]*B_bot[0,k] + A[i,3]*B_bot[1,k]
    #             = [A[i,0],A[i,1]] * B_top[:,k] + [A[i,2],A[i,3]] * B_bot[:,k]
    # 
    # For fixed i, this is: 1x2 * 2x4 + 1x2 * 2x4 = 1x4 + 1x4
    # Each 1x2*2x4 at schoolbook rank 8. Total per i: 16. For 2 i's: 32.
    # 
    # But we can use Strassen on 2x2*2x2 blocks!
    # For 1x2*2x4: split the 2x4 into two 2x2 blocks.
    # 1x2*2x2 at schoolbook is 4, or we can view differently.
    
    # OK let me just go with rank 40: 2x4x4 schoolbook (32) + 2x4x2 schoolbook (8) = 40.
    # That beats 42.
    
    # Actually, even simpler: just do full schoolbook but in a structured way.
    # 2*4*6 = 48. No, that's worse.
    
    # Let me do: 2x4x4 at 32 + 2x4x2 at 8 = 40.
    
    # Hmm, but I want to try for 37. Let me implement the 2x2x4 at rank 14 approach.
    
    # For 2x2x4: A is 2x2, B is 2x4, C = A*B is 2x4.
    # Split B = [B1 B2] where B1, B2 are 2x2.
    # C = [A*B1, A*B2].
    # Each 2x2*2x2 at Strassen rank 7.
    # Total: 14 terms.
    
    # For our problem:
    # C[i,k] for i in {0,1}, k in {0,1,2,3}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # 
    # Group: C[i,0:2] = [A[i,0],A[i,1]] * B[0:2,0:2] + [A[i,2],A[i,3]] * B[2:4,0:2]
    # C[i,2:4] = [A[i,0],A[i,1]] * B[0:2,2:4] + [A[i,2],A[i,3]] * B[2:4,2:4]
    # 
    # For each i, we have four 1x2*2x2 products.
    # 1x2*2x2: x = [x0,x1], Y = [[y00,y01],[y10,y11]], result = [x0*y00+x1*y10, x0*y01+x1*y11]
    # Schoolbook: 4 terms. Can we do it in 3? 
    # r1 = x0*y00, r2 = x1*y10, r3 = x0*y01, r4 = x1*y11. That's 4.
    # Or: r1=(x0+x1)(y00+y10), r2=x0*y00, r3=x1*y10, r4=(x0+x1)(y01+y11), r5=x0*y01, r6=x1*y11
    # Result[0]=r1-r2-r3+r? No, this doesn't simplify nicely for 1x2*2x2.
    # Minimum rank for 1x2*2x2 is 4 (it's just two independent dot products of length 2).
    # 
    # So each 1x2*2x2 needs 4 terms. Four of them per i: 16. Two i's: 32.
    # Plus 2x4x2 at 8. Total 40.
    
    # Let me just implement rank 40.
    
    # Actually, I realize I should just carefully construct the schoolbook 2x4x6 = 48
    # and see if I can merge any terms. But 48 > 42, so that won't help.
    
    # Let me go with 40: structured schoolbook split.
    
    # Terms for C_left (2x4):
    # For i in {0,1}, k in {0,1,2,3}:
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # Each term (i,j,k) contributes to U[i*4+j, t], V[j*6+k, t], W[k*2+i, t]
    # Total: 2*4*4 = 32 terms.
    
    # Terms for C_right (2x2):
    # For i in {0,1}, k in {0,1} (actual k = 4+k):
    #   C[i,k] = A[i,0]*B[0,4+k] + A[i,1]*B[1,4+k] + A[i,2]*B[2,4+k] + A[i,3]*B[3,4+k]
    # Total: 2*4*2 = 8 terms.
    
    # Grand total: 40 terms.
    
    col = 0
    # Part 1: C_left, k in 0..3
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Part 2: C_right, k in 4..5
    for i in range(2):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
