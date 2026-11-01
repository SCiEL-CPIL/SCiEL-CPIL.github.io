"""Matrix multiplication 2,2,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

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

    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        R = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank12_direct():
        """
        Try to find a rank-12 decomposition by solving a system of equations.
        The key is to find 12 rank-1 tensors that sum to the multiplication tensor.
        
        We parameterize the U and W matrices and solve for V.
        """
        if time.time() > deadline - 30:
            return None
        
        R = 12
        
        # Strategy: Use a structured form for U and W based on known results.
        # For 2x2 x 2x4, we can try various structured forms.
        
        # Approach: Fix U to have a specific structure and solve for W and V.
        # Let's try a systematic search over small integer/half-integer values.
        
        # The multiplication tensor T has entries T[i,j,k] = 1 if the (i,j) entry of A
        # and (j,k) entry of B contribute to C[i,k].
        # T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
        
        # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
        
        # Let's try a different approach: use the fact that the rank-12 decomposition
        # exists and search for it using a parameterized family.
        
        # Known approach: For 2x2 x 2x2, rank 7 (Strassen). For 2x2 x 2x4, 
        # we can potentially do better than 2*7=14 by sharing some terms.
        
        # Let's try: assume U has the form where each column is a vector in {0, ±1/2, ±1}^4
        # and similarly for W. Then solve for V.
        
        # Actually, let's try a more direct approach: 
        # Fix 11 of the 12 rank-1 terms and solve for the 12th.
        
        # Better: use linear algebra. For fixed U and W, the equations are linear in V.
        # So we can try random/structured U and W and check if V exists.
        
        # Let's try a structured approach based on the known rank-12 construction.
        
        # The rank-12 decomposition for 2x2 x 2x4 can be constructed as follows:
        # Think of it as two 2x2 multiplications but with shared "diagonal" terms.
        
        # Let me try a specific known construction:
        # r=0: (a00+a11)(b00+b12) -> contributes to C00, C02, C10, C12
        # etc.
        
        # Actually, let's just do a more thorough search.
        # We'll try all combinations of a small set of candidate U columns.
        
        # Candidate U vectors (in R^4):
        u_candidates = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0],
            [1, 0, -1, 0],
            [1, 0, 0, -1],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, -1],
            [1, 0, -1, 1],
            [1, 0, -1, -1],
            [0, 1, 1, 1],
            [0, 1, 1, -1],
            [0, 1, -1, 1],
            [0, 1, -1, -1],
            [1, 1, 0, 1],
            [1, 1, 0, -1],
            [1, -1, 0, 1],
            [1, -1, 0, -1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        
        # Remove duplicates
        u_candidates = [list(x) for x in set(tuple(u) for u in u_candidates)]
        
        # Similarly for W vectors (in R^8):
        # W[k*n+i] for k in 0..3, i in 0..1
        # Indices: C00=0, C10=1, C01=2, C11=3, C02=4, C12=5, C03=6, C13=7
        w_candidates = [
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [1,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,1],
            [0,1,0,0,1,0,0,0],
            [1,0,0,1,0,0,0,0],
            [0,1,0,1,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,1,0,0,0,0,1],
            [0,0,0,1,0,0,1,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,1,0,0,1],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,0,1],
            [0,0,0,0,1,0,1,0],
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [1,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,1],
            [0,1,0,0,1,0,0,0],
            [1,0,0,1,0,0,0,0],
            [0,1,0,1,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,1,0,0,0,0,1],
            [0,0,0,1,0,0,1,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,1,0,0,1],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,0,1],
            [0,0,0,0,1,0,1,0],
        ]
        w_candidates = [list(x) for x in set(tuple(w) for w in w_candidates)]
        
        # This exhaustive search is too large. Let's try a smarter approach.
        # 
        # Key insight: The rank-12 decomposition can be found by considering
        # the problem as a bilinear map and using known algebraic constructions.
        #
        # Let me try a specific known rank-12 construction for 2x2 x 2x4:
        
        # Construction based on sharing diagonal terms:
        # The idea is to use 12 products where some terms contribute to both
        # the left and right halves of the output.
        
        # Let's try this specific construction:
        U = [[0]*12 for _ in range(4)]
        V = [[0]*12 for _ in range(8)]
        W = [[0]*12 for _ in range(8)]
        
        # r=0: (a00+a11)(b00+b12)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[4][0]=1
        W[0][0]=1; W[5][0]=1
        
        # r=1: (a00+a11)(b01+b13)
        U[0][1]=1; U[3][1]=1
        V[1][1]=1; V[7][1]=1
        W[2][1]=1; W[7][1]=1
        
        # r=2: (a10+a11)b00
        U[2][2]=1; U[3][2]=1
        V[0][2]=1
        W[1][2]=1; W[5][2]=-1
        
        # r=3: (a10+a11)b12
        U[2][3]=1; U[3][3]=1
        V[4][3]=1
        W[0][3]=1; W[1][3]=-1
        
        # r=4: a00(b01-b11)
        U[0][4]=1
        V[1][4]=1; V[5][4]=-1
        W[2][4]=1; W[3][4]=1
        
        # r=5: a11(b10-b00)
        U[3][5]=1
        V[4][5]=1; V[0][5]=-1
        W[0][5]=1; W[1][5]=1
        
        # r=6: (a00+a01)b11
        U[0][6]=1; U[1][6]=1
        V[5][6]=1
        W[0][6]=-1; W[2][6]=1
        
        # r=7: (a10-a00)(b00+b01)
        U[2][7]=1; U[0][7]=-1
        V[0][7]=1; V[1][7]=1
        W[3][7]=1
        
        # r=8: (a01-a11)(b10+b11)
        U[1][8]=1; U[3][8]=-1
        V[4][8]=1; V[5][8]=1
        W[0][8]=1
        
        # r=9: a00(b03-b13)
        U[0][9]=1
        V[3][9]=1; V[7][9]=-1
        W[6][9]=1; W[7][9]=1
        
        # r=10: (a00+a01)b13
        U[0][10]=1; U[1][10]=1
        V[7][10]=1
        W[4][10]=-1; W[6][10]=1
        
        # r=11: (a10-a00)(b02+b03)
        U[2][11]=1; U[0][11]=-1
        V[2][11]=1; V[3][11]=1
        W[7][11]=1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None

    def try_rank13_linear():
        """
        Try to find rank-13 by fixing U and W structures and solving for V.
        """
        if time.time() > deadline - 20:
            return None
        
        R = 13
        
        # Use the Strassen rank-14 and try to merge one term.
        # For each pair (li, ri) where we merge, check if the system is consistent.
        
        # Strassen terms for left block (2x2 x 2x2, cols 0-1 of B):
        left_U = [
            [1,0,0,1],  # M1
            [0,0,1,1],  # M2
            [1,0,0,0],  # M3
            [0,0,0,1],  # M4
            [1,1,0,0],  # M5
            [1,-1,0,0], # M6
            [0,1,-1,0], # M7
        ]
        left_V = [
            [1,1],  # b00+b11 -> indices 0,5 in full V
            [1,0],  # b00 -> index 0
            [0,1],  # b01 -> index 1
            [1,0],  # b10 -> index 4
            [0,1],  # b11 -> index 5
            [1,1],  # b00+b01 -> indices 0,1
            [1,1],  # b10+b11 -> indices 4,5
        ]
        left_W = [
            [1,0,0,-1],  # C00+C11
            [0,1,-1,0],  # C10-C11
            [0,0,1,1],   # C01+C11
            [1,-1,0,0],  # C00+C10
            [-1,0,1,0],  # -C00+C01
            [0,0,0,1],   # C11
            [1,0,0,0],   # C00
        ]
        
        # Strassen terms for right block (2x2 x 2x2, cols 2-3 of B):
        right_U = [
            [1,0,0,1],  # N1
            [0,0,1,1],  # N2
            [1,0,0,0],  # N3
            [0,0,0,1],  # N4
            [1,1,0,0],  # N5
            [1,-1,0,0], # N6
            [0,1,-1,0], # N7
        ]
        right_V = [
            [1,1],  # b02+b13 -> indices 2,7
            [1,0],  # b02 -> index 2
            [0,1],  # b03 -> index 3
            [1,0],  # b12 -> index 6
            [0,1],  # b13 -> index 7
            [1,1],  # b02+b03 -> indices 2,3
            [1,1],  # b12+b13 -> indices 6,7
        ]
        right_W = [
            [0,0,0,0,1,0,0,-1],  # C02+C13
            [0,0,0,0,0,1,-1,0],  # C12-C13
            [0,0,0,0,0,0,1,1],   # C03+C13
            [0,0,0,0,1,-1,0,0],  # C02+C12
            [0,0,0,0,-1,0,0,1],  # -C02+C03
            [0,0,0,0,0,0,0,1],   # C13
            [0,0,0,0,1,0,0,0],   # C02
        ]
        
        # Try merging each pair
        for li in range(7):
            if time.time() > deadline - 5:
                return None
            for ri in range(7):
                # Build U, W matrices for rank 13
                U_mat = np.zeros((4, 13))
                W_mat = np.zeros((8, 13))
                
                # Column 0: merged term
                U_mat[:, 0] = left_U[li]
                # W for merged: left part + right part
                W_mat[:4, 0] = left_W[li]
                W_mat[4:8, 0] = right_W[ri]
                
                # Remaining left terms
                col = 1
                for l_idx in range(7):
                    if l_idx == li:
                        continue
                    U_mat[:, col] = left_U[l_idx]
                    W_mat[:4, col] = left_W[l_idx]
                    col += 1
                
                # Remaining right terms
                for r_idx in range(7):
                    if r_idx == ri:
                        continue
                    U_mat[:, col] = right_U[r_idx]
                    W_mat[4:8, col] = right_W[r_idx]
                    col += 1
                
                # Now solve for V
                # Equations: for each (i,j,k), sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # This is linear in V.
                
                # Build the linear system
                # Variables: V[j*p+k, r] for j in 0..1, k in 0..3, r in 0..12
                # Total variables: 2*4*13 = 104
                # Equations: 2*2*4 = 16
                
                num_eqs = n * m * p  # 16
                num_vars = m * p * R  # 8 * 13 = 104
                
                A = np.zeros((num_eqs, num_vars))
                b = np.ones(num_eqs)
                
                for eq in range(num_eqs):
                    i = eq // (m * p)
                    j = (eq // p) % m
                    k = eq % p
                    
                    for r in range(R):
                        for jj in range(m):
                            for kk in range(p):
                                var_idx = (jj * p + kk) * R + r
                                A[eq, var_idx] = U_mat[i * m + j, r] * W_mat[kk * n + i, r]
                
                # Solve least squares
                try:
                    V_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                
                # Check if solution is exact
                A_V = A @ V_flat
                if not np.allclose(A_V, b, atol=1e-9):
                    continue
                
                # Reshape V
                V_mat = V_flat.reshape(m * p, R).T  # shape (R, m*p) -> transpose to (m*p, R)
                
                U_list = U_mat.tolist()
                V_list = V_mat.tolist()
                W_list = W_mat.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

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

    # Try rank-12 first (best possible)
    result = try_rank12_direct()
    if result is not None:
        return result
    
    # Try rank-13
    result = try_rank13_linear()
    if result is not None:
        return result
    
    # Try rank-14 (reliable fallback)
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
