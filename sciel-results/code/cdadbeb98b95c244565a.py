"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Use Strassen-style decomposition:
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    #
    # For each 2x2 * 2x5 product, use a rank-7 Strassen decomposition.
    # Total rank = 7 + 7 = 14... but that's the bilinear rank for the 
    # sub-problems. Let me construct it explicitly.
    #
    # Actually, for 2x2 * 2x5:
    # A = [[a,b],[c,d]], B = [[e,f,g,h,i],[j,k,l,m,o]]
    # C = [[ae+bj, af+bk, ag+bl, ah+bm, ai+bo],
    #      [ce+dj, cf+dk, cg+dl, ch+dm, ci+do]]
    #
    # Strassen for 2x2*2x2 (per column):
    # M1 = (a+d)(e+j)
    # M2 = (c+d)e
    # M3 = a(f-j)
    # M4 = d(j-e)
    # M5 = (a+b)j
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(j+k)  -- wait, this doesn't work for non-square
    #
    # Let me just use the known rank-36 construction directly.
    # The 4x4 middle split with Strassen gives:
    # 7 multiplications for the "upper" 2x2 block products and 7 for "lower"
    # but with the 5 columns, each "multiplication" is actually a dot product
    # of length 2 with each of 5 columns... 
    #
    # Let me think differently. The tensor T has shape (8, 20, 10).
    # T[i, j, k] = 1 if A_elem[i] * B_elem[j] contributes to C_elem[k].
    #
    # i indexes A (2*4=8): i = row*4 + col, row in [0,1], col in [0,3]
    # j indexes B (4*5=20): j = row*5 + col, row in [0,3], col in [0,4]
    # k indexes C (2*5=10): k = row*5 + col, row in [0,1], col in [0,4]
    #
    # A[row, col] * B[row2, col2] contributes to C[row, col2] if col == row2.
    # So T[i, j, k] = 1 iff:
    #   i = row*4 + col_a, j = row_b*5 + col_c, k = row*5 + col_c
    #   where col_a == row_b
    #
    # Let me build the rank-36 decomposition using Strassen on the 4-dim split.
    
    # Split the 4 middle indices into two groups of 2: {0,1} and {2,3}
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # B1 = B[0:2, :] (2x5), B2 = B[2:4, :] (2x5)
    # C = A1*B1 + A2*B2
    
    # For 2x2 * 2x5, we need a bilinear decomposition.
    # The rank of 2x2 * 2x5 matrix multiplication tensor:
    # Each of the 5 columns is a 2x2 * 2x1 problem (rank 2 each).
    # But we can share across columns using Strassen-like identities.
    # 
    # For 2x2 * 2x1: A=[[a,b],[c,d]], B=[[e],[f]], C=[[ae+bf],[ce+df]]
    # Rank 2: just the two multiplications (a*e, b*f for first row, etc.)
    # Actually rank is 2: ae+bf and ce+df. Can we do it in rank < 4? 
    # Yes, it's rank 2 (two independent bilinear forms).
    #
    # For 2x2 * 2x5: 5 independent 2x2*2x1 problems, total rank 10 if no sharing.
    # But with Strassen on the 2x2 structure, we can do 7 "shared" multiplications
    # where each produces a 1x5 vector? No, that doesn't make sense dimensionally.
    #
    # Let me just explicitly construct a valid decomposition.
    # I'll use the approach: for each of the 4 "middle" indices j, and for each
    # of the 2 row indices i, and each of the 5 col indices k, we have a term.
    # That's 4*2*5 = 40 terms (schoolbook).
    #
    # To reduce to 36, I need to find 4 linear dependencies.
    # 
    # Let me try a different approach: use the fact that we can combine
    # some terms. For example, for a fixed middle index j, the contribution
    # to C[i,k] is A[i,j]*B[j,k]. If we can express some combination of
    # these as a product of sums, we save rank.
    
    # Let me construct the rank-36 solution explicitly.
    # Based on the knowledge that it works, let me build it.
    
    # Approach: Use Strassen on splitting the 4-dim into 2+2.
    # A = [A1, A2], B = [B1; B2]
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x5):
    # A1 = [[a00, a01], [a10, a11]]
    # B1 = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]]
    # C1 = A1*B1 (2x5)
    #
    # Using Strassen-like 7-term decomposition for 2x2 * 2x5:
    # M1 = (a00+a11) * (B1[0,:] + B1[1,:])  -- 1x5 vector
    # M2 = (a10+a11) * B1[0,:]               -- 1x5 vector  
    # M3 = a00 * (B1[1,:] - B1[0,:])          -- 1x5 vector
    # M4 = a11 * (B1[1,:] - B1[0,:])          -- 1x5 vector
    # M5 = (a00+a01) * B1[1,:]                -- 1x5 vector
    # M6 = (a10-a00) * (B1[0,:] + B1[1,:])    -- 1x5 vector
    # M7 = (a01-a11) * (B1[0,:] + B1[1,:])    -- 1x5 vector
    #
    # Wait, this doesn't give the right C1. Let me use the standard Strassen:
    # C1[0,0] = M1 + M4 - M5 + M7
    # C1[0,1] = M3 + M5
    # C1[1,0] = M2 + M4
    # C1[1,1] = M1 - M2 + M3 + M7
    #
    # But this is for 2x2 * 2x2. For 2x2 * 2x5, each "element" of B is a 1x5 row.
    # So the same formulas work element-wise across the 5 columns!
    #
    # So for A1*B1, we have 7 "multiplications" where each is a scalar * (1x5 vector).
    # In the tensor decomposition, each such term corresponds to 5 tensor entries
    # (one per column k). So each Strassen term contributes rank 5?
    # No, each Strassen term is a single bilinear form: scalar_A * scalar_B where
    # scalar_B is actually a vector. In the tensor, this means for each k,
    # we have a term U[i,j]*V[j,k]*W[k,i] where V[j,k] varies with k.
    #
    # I think the correct interpretation is: each Strassen "multiplication" M_r
    # corresponds to one rank-1 term in the tensor, but since it involves
    # products with all 5 columns, it actually spans 5 tensor entries.
    #
    # Let me just build it explicitly. For the full 2x4*4x5:
    # 
    # I'll create 14 rank-1 terms (7 for A1*B1, 7 for A2*B2).
    # But wait, each "term" in Strassen for 2x2*2x5 involves a product of
    # a linear combination of A elements with a linear combination of B elements
    # (which are vectors of length 5). So in the tensor, each such term
    # contributes to 5 different (i,k) pairs for each j... 
    #
    # I'm overcomplicating this. Let me just directly verify a rank-36 solution.
    
    # Direct construction: schoolbook is rank 40. Let me try to find 4
    # linear combinations that reduce the rank.
    
    # Actually, let me use a simpler approach. The known result for 2x4*4x5
    # has rank 36. Let me construct it by using Strassen on the 4x4 block
    # structure properly.
    
    # The key: A (2x4) times B (4x5). Think of it as:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Split j into j1 in {0,1} and j2 in {2,3}:
    # C[i,k] = sum_{j1=0}^{1} A[i,j1]*B[j1,k] + sum_{j2=0}^{1} A[i,j2+2]*B[j2+2,k]
    #
    # First sum: 2x2 * 2x5 (A[:, 0:2] * B[0:2, :])
    # Second sum: 2x2 * 2x5 (A[:, 2:4] * B[2:4, :])
    #
    # For each 2x2 * 2x5, the tensor rank is 7 (using Strassen generalized).
    # Total: 7 + 7 = 14? That can't be right for rank 36.
    #
    # I think the issue is that the "rank" in the Strassen sense for 2x2*2x5
    # is not 7. Let me reconsider.
    #
    # For 2x2 * 2x2: tensor shape (4,4,4), rank 7.
    # For 2x2 * 2x5: tensor shape (4, 10, 10). The rank should be 5*2 = 10
    # (each of the 5 columns is an independent 2x2*2x1 problem of rank 2).
    # Or can it be less? I believe it's 10.
    #
    # So total rank = 10 + 10 = 20? Still not 36.
    #
    # I think I'm confusing myself. Let me just go with the schoolbook rank 40
    # and try to find a valid lower-rank solution through search.
    
    # Schoolbook decomposition (rank 40)
    rank_school = n * m * p  # 40
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find rank 36 by searching for 4 linear dependencies
    # This is complex, so let me return schoolbook for now
    # and rely on the known result that 36 is achievable.
    
    # Actually, let me construct the rank-36 solution properly.
    # Based on literature, for 2x4*4x5, rank 36 is achievable.
    
    # Let me try a structured approach: use the fact that some products
    # can be shared. I'll use a greedy search for linear dependencies.
    
    # For now, return schoolbook as a valid solution.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
