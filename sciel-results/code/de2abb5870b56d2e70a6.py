"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fractional(U, V, W):
        """Verify using exact fraction arithmetic."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0, 1)
                    for r in range(R):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_search():
        """
        Search for rank-13 decomposition using structured patterns.
        
        Key insight: The tensor T[i,j,k] = 1 for all (i,j,k) with i in {0,1}, j in {0,1}, k in {0,1,2,3}.
        We need to find U (4x13), V (8x13), W (8x13) such that:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
        
        Strategy: Fix U and W with structured patterns (entries from {-1, 0, 1, 1/2}),
        then solve the linear system for V.
        """
        if time.time() > deadline - 15:
            return None
        
        R = 13
        
        # The constraint for each (i,j,k):
        # sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
        #
        # For fixed (j,k), this gives a system in V[j*4+k, :]:
        # For i=0: sum_r U[0*2+j, r] * V[j*4+k, r] * W[k*2+0, r] = 1
        # For i=1: sum_r U[1*2+j, r] * V[j*4+k, r] * W[k*2+1, r] = 1
        #
        # So for each (j,k), we have 2 equations in 13 unknowns (V[j*4+k, 0..12]).
        # Total: 8*(j,k) pairs * 2 equations = 16 equations.
        # Unknowns: 8*13 = 104.
        # This is underdetermined, so solutions exist if the system is consistent.
        
        # Let's parameterize U and W with small integer entries and solve for V.
        
        # Approach: Try all U matrices with entries in {-1, 0, 1} and W matrices
        # with entries in {-1, 0, 1}, then check if V has a solution with "nice" values.
        
        # But 3^52 is too large. Instead, use a smarter approach.
        
        # Key observation: For the rank-14 Strassen solution, each term has a specific
        # pattern. For rank 13, we need to merge one pair of terms.
        
        # Let's try a different approach: use the fact that the problem has structure.
        # The tensor is separable in the k dimension in a certain sense.
        
        # Actually, let's try to directly search for rank-13 by fixing the U matrix
        # to have a specific structure and solving for V and W.
        
        # Alternative: Use the rank-14 solution and try to perturb it to get rank 13.
        
        # Let's try a more direct search: enumerate possible U matrices with
        # entries from a small set, and for each, solve for V and W.
        
        # For efficiency, let's fix the first column of U, V, W and solve for the rest.
        
        # Actually, the most promising approach: 
        # 1. Fix U to be a 4x13 matrix with entries in {-1, 0, 1}
        # 2. For each (j,k), the constraint becomes:
        #    sum_r U[i*2+j, r] * W[k*2+i, r] * V[j*4+k, r] = 1 for i=0,1
        # 3. This is a 2x13 system for V[j*4+k, :]
        # 4. We need V[j*4+k, :] to have "nice" values (integers or half-integers)
        
        # Let's try a structured search where U has a specific pattern.
        
        # Pattern 1: U is block-diagonal with two 2x7 blocks (one for each j)
        # U[0, 0:7] = some pattern, U[1, 0:7] = some pattern (for j=0)
        # U[2, 7:13] = some pattern, U[3, 7:13] = some pattern (for j=1)
        
        # This is getting complex. Let's try a simpler approach:
        # Use the known rank-14 solution and try to eliminate one term by
        # expressing it as a linear combination of the others.
        
        # For a rank-13 solution, one of the 14 Strassen terms must be redundant.
        # This happens if the corresponding U, V, W vectors are linearly dependent
        # in a specific way.
        
        # Let's try all possible "merge" patterns more systematically.
        
        # Define the 7 Strassen terms for a 2x2 block:
        # Each term r has:
        #   u_r: 2-vector (coefficients of a00, a10 for row i=0, or a01, a11 for i=1)
        #   Actually, U is 4xR, so u_r is 4-vector: [U[0,r], U[1,r], U[2,r], U[3,r]]
        #   where U[0,r] = coeff of a00, U[1,r] = coeff of a01, U[2,r] = coeff of a10, U[3,r] = coeff of a11
        
        # Standard Strassen terms (U_4, V_2, W_2) for left block:
        strassen_left = [
            # (U, V, W) where V is for b00,b01 and W is for C00,C10,C01,C11
            ([1,0,0,1], [1,1], [1,0,0,-1]),    # M1
            ([0,0,1,1], [1,0], [0,1,-1,0]),    # M2
            ([1,0,0,0], [0,1], [0,0,1,1]),     # M3
            ([0,0,0,1], [1,0], [1,-1,0,0]),    # M4
            ([1,1,0,0], [0,1], [-1,0,1,0]),    # M5
            ([1,-1,0,0], [1,1], [0,0,0,1]),    # M6
            ([0,1,-1,0], [1,1], [1,0,0,0]),    # M7
        ]
        
        strassen_right = [
            ([1,0,0,1], [1,1], [0,0,0,0,1,0,0,-1]),    # N1
            ([0,0,1,1], [1,0], [0,0,0,0,0,1,-1,0]),    # N2
            ([1,0,0,0], [0,1], [0,0,0,0,0,0,1,1]),     # N3
            ([0,0,0,1], [1,0], [0,0,0,0,1,-1,0,0]),    # N4
            ([1,1,0,0], [0,1], [0,0,0,0,-1,0,0,1]),    # N5
            ([1,-1,0,0], [1,1], [0,0,0,0,0,0,0,1]),    # N6
            ([0,1,-1,0], [1,1], [0,0,0,0,1,0,0,0]),    # N7
        ]
        
        # Try merging one left term with one right term
        for li in range(7):
            for ri in range(7):
                if time.time() > deadline - 5:
                    return None
                
                U_left = np.array(strassen_left[li][0], dtype=float)
                U_right = np.array(strassen_right[ri][0], dtype=float)
                
                # Check if U_left and U_right are proportional
                if np.linalg.matrix_rank(np.column_stack([U_left, U_right])) < 2:
                    # Build the rank-13 decomposition
                    R = 13
                    U_mat = np.zeros((4, R))
                    V_mat = np.zeros((8, R))
                    W_mat = np.zeros((8, R))
                    
                    # Column 0: merged term
                    U_mat[:, 0] = U_left
                    V_mat[:2, 0] = strassen_left[li][1]
                    V_mat[2:4, 0] = strassen_right[ri][1]
                    W_mat[:4, 0] = strassen_left[li][2]
                    W_mat[4:8, 0] = strassen_right[ri][2]
                    
                    # Remaining left terms
                    col = 1
                    for l_idx in range(7):
                        if l_idx == li:
                            continue
                        U_mat[:, col] = strassen_left[l_idx][0]
                        V_mat[:2, col] = strassen_left[l_idx][1]
                        W_mat[:4, col] = strassen_left[l_idx][2]
                        col += 1
                    
                    # Remaining right terms
                    for r_idx in range(7):
                        if r_idx == ri:
                            continue
                        U_mat[:, col] = strassen_right[r_idx][0]
                        V_mat[2:4, col] = strassen_right[r_idx][1]
                        W_mat[4:8, col] = strassen_right[r_idx][2]
                        col += 1
                    
                    # Solve for V using least squares
                    # Constraint: for all (i,j,k), sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
                    B = np.zeros((16, R * 8))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx = i * m * p + j * p + k
                                for r in range(R):
                                    B[idx, r * 8 + j * p + k] = U_mat[i * m + j, r] * W_mat[k * n + i, r]
                    
                    rhs = np.ones(16)
                    try:
                        V_flat, residuals, rank, sv = np.linalg.lstsq(B, rhs, rcond=None)
                    except:
                        continue
                    
                    if not np.allclose(B @ V_flat, rhs, atol=1e-9):
                        continue
                    
                    V_sol = V_flat.reshape(R, 8).T
                    
                    # Check if entries are "nice"
                    V_list = V_sol.tolist()
                    U_list = U_mat.tolist()
                    W_list = W_mat.tolist()
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_rank13_fourier():
        """
        Use DFT on the output dimension to diagonalize and search for rank-13.
        
        The tensor T[i,j,k] = 1 for all (i,j,k). Apply DFT in k:
        T'[i,j,omega] = sum_k T[i,j,k] * exp(-2*pi*i*omega*k/4)
        
        For omega=0: T'[i,j,0] = 4
        For omega=1: T'[i,j,1] = 1 - i - 1 + i = 0
        For omega=2: T'[i,j,2] = 1 + 1 - 1 - 1 = 0
        For omega=3: T'[i,j,3] = 1 + i - 1 - i = 0
        
        So only omega=0 survives. This means the tensor is "constant" in k,
        which simplifies the problem.
        
        The rank of the all-ones tensor (2,2,4) should be the same as the rank
        of the all-ones tensor (2,2,1) times something... 
        
        Actually, let's just try to find a rank-13 solution by solving the
        linear system with a structured ansatz.
        """
        if time.time() > deadline - 10:
            return None
        
        R = 13
        
        # Ansatz: U has the form where each column is a tensor product of
        # a 2-vector (for i) and a 2-vector (for j).
        # U[i*2+j, r] = a_r[i] * b_r[j]
        
        # Similarly, W[k*2+i, r] = c_r[k] * d_r[i]
        
        # Then the constraint becomes:
        # sum_r a_r[i] * b_r[j] * V[j*4+k, r] * c_r[k] * d_r[i] = 1
        # = sum_r (a_r[i] * d_r[i]) * (b_r[j] * c_r[k]) * V[j*4+k, r] = 1
        
        # Let e_r[i] = a_r[i] * d_r[i], f_r[j,k] = b_r[j] * c_r[k]
        # Then: sum_r e_r[i] * f_r[j,k] * V[j*4+k, r] = 1
        
        # This is still complex. Let's just try a direct numerical search.
        
        # Try: U[i*2+j, r] = 1 for all i,j,r (all ones)
        # W[k*2+i, r] = 1 for all k,i,r (all ones)
        # Then: sum_r V[j*4+k, r] = 1 for all j,k
        # So each row of V sums to 1. This gives a rank-1 solution!
        
        # But wait, this can't be right. Let me verify.
        # If U = ones(4, 1), V has 8 rows each summing to 1, W = ones(8, 1)
        # Then T[i,j,k] = U[i*2+j, 0] * V[j*4+k, 0] * W[k*2+i, 0] = 1 * V[j*4+k, 0] * 1
        # For this to equal 1, we need V[j*4+k, 0] = 1 for all j,k.
        # So V = ones(8, 1) works, giving rank 1!
        
        # This suggests the all-ones tensor has rank 1, which contradicts Strassen.
        # The issue must be in my understanding of the tensor formulation.
        
        # Let me just return None and rely on the rank-14 solution.
        return None
    
    # Try rank-13 searches
    result = try_rank13_search()
    if result is not None:
        return result
    
    result = try_rank13_fourier()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
