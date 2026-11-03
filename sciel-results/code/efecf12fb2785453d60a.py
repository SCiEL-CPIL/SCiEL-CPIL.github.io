"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
from fractions import Fraction
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 60
    
    def verify_exact(U, V, W, n, m, p):
        """Check if decomposition is exact using Fraction arithmetic on random instances."""
        random.seed(42)
        for trial in range(3):
            A = [[Fraction(random.randint(-5, 5)) for _ in range(m)] for _ in range(n)]
            B = [[Fraction(random.randint(-5, 5)) for _ in range(p)] for _ in range(m)]
            C_true = [[Fraction(0) for _ in range(p)] for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_true[i][k] += A[i][j] * B[j][k]
            rank = len(U[0])
            for i in range(n):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        w_val = W[k * n + i][c]
                        if w_val != 0:
                            for j in range(m):
                                u_val = U[i * m + j][c]
                                v_val = V[j * p + k][c]
                                if u_val != 0 and v_val != 0:
                                    total += u_val * v_val * w_val
                    if total != C_true[i][k]:
                        return False
        return True

    def build_strassen_2x2x2():
        """Build a Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
        # Strassen's 7 products for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
        
        # U[i*m+j, c] where i,j in {0,1}
        # V[j*p+k, c] where j,k in {0,1}
        # W[k*n+i, c] where k,i in {0,1}
        
        # M1: (a+d)(e+h) -> U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=-1, W[0,1]=1, W[1,0]=-1
        # Wait, let me be more careful with indices.
        # For term c, the contribution to C[i,k] is sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let me define each Strassen product:
        # Product r: U_r[i,j], V_r[j,k], W_r[k,i]
        
        # M1 = (a+d)(e+h): 
        # U1: U1[0,0]=1, U1[1,1]=1 (a and d)
        # V1: V1[0,0]=1, V1[1,1]=1 (e and h)  
        # W1: W1[0,0]=1, W1[1,1]=1, W1[0,1]=-1, W1[1,0]=-1
        # Check: C[0,0] += U1[0,0]*V1[0,0]*W1[0,0] + U1[0,1]*V1[1,0]*W1[0,0] + U1[1,0]*V1[0,1]*W1[1,0] + U1[1,1]*V1[1,1]*W1[1,0]
        # = 1*1*1 + 0 + 0 + 1*1*(-1) = 0? That's wrong.
        
        # Let me reconsider. The tensor decomposition is:
        # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For Strassen, each product M_r contributes to multiple C entries.
        # M1 = (a+d)(e+h) contributes to C11 and C22.
        # Specifically: C11 += M1, C22 += M1, and we need coefficients.
        
        # Actually, let me use the standard Strassen formulation:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e  
        # p3 = a(f-h)
        # p4 = d(g-e)
        # p5 = (a+b)h
        # p6 = (c-a)(e+f)
        # p7 = (b-d)(g+h)
        
        # C11 = p1 + p4 - p5 + p7
        # C12 = p3 + p5
        # C21 = p2 + p4
        # C22 = p1 - p2 + p3 + p6
        
        # For each product p_r, we need U_r, V_r, W_r such that:
        # sum_j U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = coefficient of p_r in C[i,k]
        
        # p1 = (a+d)(e+h) = ae + ah + de + dh
        # In C11: coeff of p1 is 1
        # In C22: coeff of p1 is 1
        # So: sum_j U1[i*m+j]*V1[j*p+k]*W1[k*n+i] = 1 if (i,k)=(0,0) or (1,1), else 0
        # And U1, V1 should encode (a+d) and (e+h)
        # U1[0,0]=1, U1[1,1]=1 (selects a and d)
        # V1[0,0]=1, V1[1,1]=1 (selects e and h)
        # Then sum_j U1[i*m+j]*V1[j*p+k] for various (i,k):
        # (i=0,k=0): j=0: U1[0,0]*V1[0,0]=1*1=1; j=1: U1[0,1]*V1[1,0]=0*0=0 -> total 1
        # (i=0,k=1): j=0: U1[0,0]*V1[0,1]=1*0=0; j=1: U1[0,1]*V1[1,1]=0*1=0 -> total 0
        # (i=1,k=0): j=0: U1[1,0]*V1[0,0]=0*1=0; j=1: U1[1,1]*V1[1,0]=1*0=0 -> total 0
        # (i=1,k=1): j=0: U1[1,0]*V1[0,1]=0*0=0; j=1: U1[1,1]*V1[1,1]=1*1=1 -> total 1
        # So we need W1 such that W1[0,0]*1 + W1[0,0]*0... 
        # Actually W1[k*n+i] multiplies the sum. For (i=0,k=0): sum=1, need result=1, so W1[0*n+0]=W1[0]=1
        # For (i=1,k=1): sum=1, need result=1, so W1[1*n+1]=W1[3]=1
        # For (i=0,k=1): sum=0, need result=0, so W1[1*n+0]=W1[2] can be anything
        # For (i=1,k=0): sum=0, need result=0, so W1[0*n+1]=W1[1] can be anything
        # Set W1[0]=1, W1[3]=1, W1[1]=0, W1[2]=0
        
        # Let me just construct all 7 terms systematically.
        terms = []
        # Each term: (U_coeffs, V_coeffs, W_coeffs) where U_coeffs is dict {(i,j): val}, etc.
        
        # p1 = (a+d)(e+h)
        U1 = {(0,0):1, (1,1):1}
        V1 = {(0,0):1, (1,1):1}
        W1 = {(0,0):1, (1,1):1}  # W[k,i]
        terms.append((U1, V1, W1))
        
        # p2 = (c+d)e
        U2 = {(1,0):1, (1,1):1}  # c=A[1,0], d=A[1,1]
        V2 = {(0,0):1, (1,0):1}  # e=B[0,0], and B[1,0]=g... wait
        # p2 = (c+d)*e, so V should select only e=B[0,0]
        # U selects c=A[1,0] and d=A[1,1]
        # V selects e=B[0,0]
        # Contribution to C21: coeff 1, C22: coeff -1
        U2 = {(1,0):1, (1,1):1}
        V2 = {(0,0):1, (1,0):1}  # This gives (c+d)*(e+g), not (c+d)*e
        # Hmm, I need to be more careful.
        
        # Let me restart with a cleaner approach. I'll use the known Strassen decomposition.
        # The 7 rank-1 tensors for 2x2x2 are well-known. Let me look them up from memory:
        
        # Using indices A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}
        
        # r=0: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
        # This is (a+d)(e+h) and contributes to C[0,0] and C[1,1]
        
        # r=1: U[1,0]=1, U[1,1]=1; V[0,0]=1; W[1,0]=1, W[1,1]=-1
        # This is (c+d)*e and contributes to C[1,0] with +1 and C[1,1] with -1
        
        # r=2: U[0,0]=1; V[0,1]=1, V[1,1]=-1; W[0,1]=1
        # This is a*(f-h) and contributes to C[0,1] with +1
        
        # r=3: U[1,1]=1; V[1,0]=1, V[0,0]=-1; W[1,0]=1
        # This is d*(g-e) and contributes to C[1,0] with +1
        
        # r=4: U[0,0]=1, U[0,1]=1; V[1,1]=1; W[0,1]=1, W[0,0]=-1
        # This is (a+b)*h and contributes to C[0,1] with +1 and C[0,0] with -1
        
        # r=5: U[1,0]=1, U[0,0]=-1; V[0,0]=1, V[0,1]=1; W[1,1]=1
        # This is (c-a)*(e+f) and contributes to C[1,1] with +1
        
        # r=6: U[0,1]=1, U[1,1]=-1; V[1,0]=1, V[1,1]=1; W[0,0]=1, W[1,0]=1
        # This is (b-d)*(g+h) and contributes to C[0,0] with +1 and C[1,0] with +1
        
        # Let me verify r=0:
        # C[0,0] += sum_j U[0,j]*V[j,0]*W[0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[1,0]*W[0,0] = 1*1*1 + 0*0*1 = 1
        # C[0,1] += sum_j U[0,j]*V[j,1]*W[1,0] = U[0,0]*V[0,1]*W[1,0] + U[0,1]*V[1,1]*W[1,0] = 1*0*0 + 0*1*0 = 0
        # C[1,0] += sum_j U[1,j]*V[j,0]*W[0,1] = U[1,0]*V[0,0]*W[0,1] + U[1,1]*V[1,0]*W[0,1] = 0*1*0 + 1*0*0 = 0
        # C[1,1] += sum_j U[1,j]*V[j,1]*W[1,1] = U[1,0]*V[0,1]*W[1,1] + U[1,1]*V[1,1]*W[1,1] = 0*0*1 + 1*1*1 = 1
        # Good, r=0 contributes 1 to C[0,0] and C[1,1].
        
        # Verify r=1: U[1,0]=1, U[1,1]=1; V[0,0]=1; W[1,0]=1, W[1,1]=-1
        # C[0,0] += sum_j U[0,j]*V[j,0]*W[0,0] = 0
        # C[0,1] += sum_j U[0,j]*V[j,1]*W[1,0] = 0
        # C[1,0] += sum_j U[1,j]*V[j,0]*W[0,1] = U[1,0]*V[0,0]*W[0,1] + U[1,1]*V[1,0]*W[0,1] = 1*1*0 + 1*0*0 = 0
        # Hmm, W[0,1] should be for k=0,i=1. Let me recheck the W indexing.
        # W[k*n+i, c], so W[0*2+1]=W[1] is for k=0,i=1.
        # In my dict W[1,0] means k=1,i=0? No, I'm using W[k,i] in the dict.
        # Let me use explicit indices.
        
        # I think there's confusion. Let me just hardcode the Strassen decomposition.
        
        U_mat = [[0]*7 for _ in range(4)]  # 2x2 A matrix flattened
        V_mat = [[0]*7 for _ in range(4)]  # 2x2 B matrix flattened
        W_mat = [[0]*7 for _ in range(4)]  # 2x2 C matrix flattened
        
        # Flatten: A[i,j] -> index i*2+j, B[j,k] -> index j*2+k, C[i,k] -> index i*2+k
        # W[k*n+i] -> W[k*2+i]
        
        # r=0: (a+d)(e+h)
        U_mat[0][0] = 1  # a = A[0,0]
        U_mat[3][0] = 1  # d = A[1,1]
        V_mat[0][0] = 1  # e = B[0,0]
        V_mat[3][0] = 1  # h = B[1,1]
        W_mat[0][0] = 1  # C[0,0]: k=0,i=0
        W_mat[3][0] = 1  # C[1,1]: k=1,i=1
        
        # r=1: (c+d)*e
        U_mat[2][1] = 1  # c = A[1,0]
        U_mat[3][1] = 1  # d = A[1,1]
        V_mat[0][1] = 1  # e = B[0,0]
        W_mat[1][1] = 1  # C[1,0]: k=0,i=1
        W_mat[3][1] = -1  # C[1,1]: k=1,i=1
        
        # r=2: a*(f-h)
        U_mat[0][2] = 1  # a = A[0,0]
        V_mat[1][2] = 1  # f = B[0,1]
        V_mat[3][2] = -1  # h = B[1,1]
        W_mat[2][2] = 1  # C[0,1]: k=1,i=0
        
        # r=3: d*(g-e)
        U_mat[3][3] = 1  # d = A[1,1]
        V_mat[2][3] = 1  # g = B[1,0]
        V_mat[0][3] = -1  # e = B[0,0]
        W_mat[1][3] = 1  # C[1,0]: k=0,i=1
        
        # r=4: (a+b)*h
        U_mat[0][4] = 1  # a = A[0,0]
        U_mat[1][4] = 1  # b = A[0,1]
        V_mat[3][4] = 1  # h = B[1,1]
        W_mat[2][4] = 1  # C[0,1]: k=1,i=0
        W_mat[0][4] = -1  # C[0,0]: k=0,i=0
        
        # r=5: (c-a)*(e+f)
        U_mat[2][5] = 1  # c = A[1,0]
        U_mat[0][5] = -1  # a = A[0,0]
        V_mat[0][5] = 1  # e = B[0,0]
        V_mat[1][5] = 1  # f = B[0,1]
        W_mat[3][5] = 1  # C[1,1]: k=1,i=1
        
        # r=6: (b-d)*(g+h)
        U_mat[1][6] = 1  # b = A[0,1]
        U_mat[3][6] = -1  # d = A[1,1]
        V_mat[2][6] = 1  # g = B[1,0]
        V_mat[3][6] = 1  # h = B[1,1]
        W_mat[0][6] = 1  # C[0,0]: k=0,i=0
        W_mat[1][6] = 1  # C[1,0]: k=0,i=1
        
        return U_mat, V_mat, W_mat

    def build_strassen_2x5x6():
        """Build a Strassen-based decomposition for 2x5x6 by applying 2x2x2 Strassen to overlapping blocks."""
        # We have A: 2x5, B: 5x6, C: 2x6
        # Split the 5 middle dimension into blocks.
        # For each pair of consecutive j values (j, j+1) and pair of consecutive k values (k, k+1),
        # we can apply a 2x2x2 Strassen decomposition.
        
        # But the dimensions don't divide evenly. Let me think differently.
        # n=2, m=5, p=6.
        # We can partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}.
        # For each (j_block, k_block) pair where both have size 2, apply Strassen.
        # j_blocks of size 2: {0,1}, {2,3} (2 blocks)
        # k_blocks of size 2: {0,1}, {2,3}, {4,5} (3 blocks)
        # So 2*3 = 6 Strassen blocks, each of rank 7.
        # The remaining j=4 needs to be handled separately: 1*3 = 3 terms (schoolbook for j=4 with k in {0,1,2,3,4,5}).
        # Total rank: 6*7 + 3*1 = 42 + 3 = 45? But we also need to handle the cross terms.
        
        # Actually, this approach doesn't work directly because the Strassen decomposition 
        # for a 2x2x2 block assumes the full 2x2 submatrices, but our A and B have more columns/rows.
        
        # Let me use a different approach: apply Strassen to 2x2x2 sub-problems.
        # For i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}:
        # The sub-problem is multiplying A[i, j0:j0+2] (2x1 vector) by B[j0:j0+2, k0:k0+2] (2x2 matrix).
        # This isn't quite 2x2x2.
        
        # Actually, the 2x2x2 Strassen applies when we have 2x2 A, 2x2 B, 2x2 C.
        # In our case, A is 2x5, so for a fixed pair of j's, A[:, j0:j0+2] is 2x2.
        # B[j0:j0+2, k0:k0+2] is 2x2.
        # C[:, k0:k0+2] is 2x2.
        # So yes, we can apply 2x2x2 Strassen for each (j_pair, k_pair).
        
        # j_pairs: (0,1), (2,3), (4,?) - j=4 is alone, so only 2 full pairs.
        # k_pairs: (0,1), (2,3), (4,5) - 3 full pairs.
        # For j=4, we handle separately with schoolbook: 1*2*3 = 6 terms (j=4, i in {0,1}, k in {0,1,2,3,4,5}).
        
        # Wait, but the Strassen decomposition for a 2x2x2 block gives 7 terms that compute
        # the full 2x2 product. If we apply it to A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2],
        # we get C[:, k0:k0+2] = A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2].
        # But the full C is the sum over all j, so we need to sum contributions from all j_pairs.
        
        # The issue is that the Strassen decomposition computes the product of the 2x2 submatrices,
        # which is correct for those entries. But we need to make sure the terms don't interfere.
        
        # Let me just build it and verify.
        
        U = [[0]*100 for _ in range(n*m)]  # 10 rows
        V = [[0]*100 for _ in range(m*p)]  # 30 rows
        W = [[0]*100 for _ in range(p*n)]  # 12 rows
        
        col = 0
        j_pairs = [(0,1), (2,3)]
        k_pairs = [(0,1), (2,3), (4,5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # Apply 2x2x2 Strassen for A[:, j0:j1+1] * B[j0:j1+1, k0:k1+1]
                # This computes C[:, k0:k1+1] contribution from these j's.
                # The 2x2x2 Strassen has 7 terms.
                # Map local indices (i_local, j_local, k_local) to global:
                # i_local -> i (0 or 1)
                # j_local -> j0 + j_local
                # k_local -> k0 + k_local
                
                U_sub, V_sub, W_sub = build_strassen_2x2x2()
                for c in range(7):
                    for i_local in range(2):
                        for j_local in range(2):
                            val = U_sub[i_local*2 + j_local][c]
                            if val != 0:
                                U[i_local * m + (j0 + j_local)][col] = val
                    for j_local in range(2):
                        for k_local in range(2):
                            val = V_sub[j_local*2 + k_local][c]
                            if val != 0:
                                V[(j0 + j_local) * p + (k0 + k_local)][col] = val
                    for k_local in range(2):
                        for i_local in range(2):
                            val = W_sub[k_local*2 + i_local][c]
                            if val != 0:
                                W[(k0 + k_local) * n + i_local][col] = val
                    col += 1
        
        # Handle j=4 separately (schoolbook)
        for i in range(n):
            for k in range(p):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # Trim to actual rank
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
        return U, V, W, col

    # Build the Strassen-based decomposition
    U, V, W, rank = build_strassen_2x5x6()
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Verify the Strassen-based decomposition
    if not verify_exact(U, V, W, n, m, p):
        # Fall back to schoolbook
        U = [[0]*rank_school for _ in range(n*m)]
        V = [[0]*rank_school for _ in range(m*p)]
        W = [[0]*rank_school for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_rank = rank_school
        best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by removing redundant columns
    def try_remove_column(U, V, W, n, m, p, col_to_remove, deadline):
        """Try removing a specific column and verify."""
        if time.time() > deadline:
            return None
        rank = len(U[0])
        U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
        V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
        W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
        if verify_exact(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
        return None
    
    # Iteratively try to remove columns
    current_U, current_V, current_W = best_U, best_V, best_W
    current_rank = best_rank
    
    max_reductions = 30
    for _ in range(max_reductions):
        if time.time() > deadline:
            break
        removed = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline:
                break
            result = try_remove_column(current_U, current_V, current_W, n, m, p, col_to_remove, deadline)
            if result is not None:
                current_U, current_V, current_W = result
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U, current_V, current_W
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
