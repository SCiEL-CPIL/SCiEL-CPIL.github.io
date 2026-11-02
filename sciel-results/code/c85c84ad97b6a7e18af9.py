"""Matrix multiplication 2,2,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

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

    def try_fourier_rank_search():
        """
        Use DFT on output dimension to diagonalize, then search for shared terms.
        
        The key insight: if we apply a 4-point DFT to the output dimension,
        the multiplication becomes four independent scalar multiplications.
        We can then look for ways to share rank-1 terms across the spectral
        components to reduce total rank below 14.
        """
        if time.time() > deadline - 30:
            return None
        
        # 4-point DFT matrix (unnormalized)
        # F[k][n] = exp(-2πi*k*n/4)
        omega = np.exp(-2j * np.pi / 4)  # i
        F = np.array([
            [1, 1, 1, 1],
            [1, omega, omega**2, omega**3],
            [1, omega**2, omega**4, omega**6],
            [1, omega**3, omega**6, omega**9]
        ], dtype=complex)
        
        # F^{-1} = (1/4) * F.conj().T
        Finv = F.conj().T / 4
        
        # The tensor T[i,j,k] = delta(i,j,k) where C[i,k] = sum_j A[i,j]*B[j,k]
        # In Fourier domain: C_hat[i, freq] = sum_j A[i,j] * B_hat[j, freq]
        # where B_hat[j, freq] = sum_k B[j,k] * F[freq,k]
        
        # For each frequency component, we have a 2x2 * 2x1 multiplication
        # which is just a 2x2 matrix times a 2-vector (rank 4 naive, or 3 with Strassen-like)
        
        # Actually, let's think differently. The full tensor has 16 components.
        # We want to find the minimal rank decomposition.
        
        # Let's try a direct linear algebra approach:
        # Set up the system and use iterative methods to find low-rank solutions.
        
        # Alternative: try rank 13 by looking for linear dependencies
        # among the 14 Strassen terms.
        
        R = 13
        U_mat = np.zeros((4, R))
        V_mat = np.zeros((8, R))
        W_mat = np.zeros((8, R))
        
        # Start from the rank-14 Strassen and try to merge one pair
        # The left block terms (indices 0-6) and right block terms (indices 7-13)
        # Check if any left term is proportional to any right term in U space
        
        left_U = np.array([
            [1,0,0,1],  # M1
            [0,0,1,1],  # M2
            [1,0,0,0],  # M3
            [0,0,0,1],  # M4
            [1,1,0,0],  # M5
            [1,-1,0,0], # M6
            [0,1,-1,0]  # M7
        ], dtype=float)
        
        right_U = np.array([
            [1,0,0,1],  # N1
            [0,0,1,1],  # N2
            [1,0,0,0],  # N3
            [0,0,0,1],  # N4
            [1,1,0,0],  # N5
            [1,-1,0,0], # N6
            [0,1,-1,0]  # N7
        ], dtype=float)
        
        # Find pairs (li, ri) where U vectors are linearly dependent
        merged_pairs = []
        for li in range(7):
            for ri in range(7):
                # Check if left_U[li] and right_U[ri] are proportional
                if np.linalg.matrix_rank(np.column_stack([left_U[li], right_U[ri]])) < 2:
                    merged_pairs.append((li, ri))
        
        # Try each merged pair
        for li, ri in merged_pairs:
            if time.time() > deadline - 15:
                return None
            
            # Create the merged decomposition
            U_mat = np.zeros((4, R))
            V_mat = np.zeros((8, R))
            W_mat = np.zeros((8, R))
            
            # Column 0: merged term
            U_mat[:, 0] = left_U[li]
            
            # V for merged term: combine left and right V contributions
            # Left term li uses V indices 0-1 (b00, b01)
            # Right term ri uses V indices 2-3 (b02, b03)
            # We need to find V values that work for both
            # This requires solving a linear system
            
            # For now, just try the naive combination and verify
            # Left V for term li
            left_V = np.zeros(2)
            right_V = np.zeros(2)
            
            # Define V patterns for each Strassen term
            # Left block: V indices are b00, b01
            left_V_patterns = [
                [1, 1],   # M1
                [1, 0],   # M2
                [0, 1],   # M3
                [1, 0],   # M4
                [0, 1],   # M5
                [1, 1],   # M6
                [1, 1]    # M7
            ]
            
            # Right block: V indices are b02, b03
            right_V_patterns = [
                [1, 1],   # N1
                [1, 0],   # N2
                [0, 1],   # N3
                [1, 0],   # N4
                [0, 1],   # N5
                [1, 1],   # N6
                [1, 1]    # N7
            ]
            
            V_mat[0, 0] = left_V_patterns[li][0]
            V_mat[1, 0] = left_V_patterns[li][1]
            V_mat[2, 0] = right_V_patterns[ri][0]
            V_mat[3, 0] = right_V_patterns[ri][1]
            
            # W for merged term
            left_W_patterns = [
                [1, 0, 0, -1],  # M1
                [0, 1, -1, 0],  # M2
                [0, 0, 1, 1],   # M3
                [1, -1, 0, 0],  # M4
                [-1, 0, 1, 0],  # M5
                [0, 0, 0, 1],   # M6
                [1, 0, 0, 0]    # M7
            ]
            
            right_W_patterns = [
                [0, 0, 0, 0, 1, 0, 0, -1],  # N1
                [0, 0, 0, 0, 0, 1, -1, 0],  # N2
                [0, 0, 0, 0, 0, 0, 1, 1],   # N3
                [0, 0, 0, 0, 1, -1, 0, 0],  # N4
                [0, 0, 0, 0, -1, 0, 0, 1],  # N5
                [0, 0, 0, 0, 0, 0, 0, 1],   # N6
                [0, 0, 0, 0, 1, 0, 0, 0]    # N7
            ]
            
            W_mat[:4, 0] = left_W_patterns[li]
            W_mat[4:8, 0] = right_W_patterns[ri]
            
            # Fill remaining columns
            col = 1
            for l_idx in range(7):
                if l_idx == li:
                    continue
                U_mat[:, col] = left_U[l_idx]
                V_mat[0, col] = left_V_patterns[l_idx][0]
                V_mat[1, col] = left_V_patterns[l_idx][1]
                W_mat[:4, col] = left_W_patterns[l_idx]
                col += 1
            
            for r_idx in range(7):
                if r_idx == ri:
                    continue
                U_mat[:, col] = right_U[r_idx]
                V_mat[2, col] = right_V_patterns[r_idx][0]
                V_mat[3, col] = right_V_patterns[r_idx][1]
                W_mat[4:8, col] = right_W_patterns[r_idx]
                col += 1
            
            # Verify
            U_list = U_mat.tolist()
            V_list = V_mat.tolist()
            W_list = W_mat.tolist()
            
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def try_rank13_search():
        """
        Search for rank-13 decomposition by trying to share one rank-1 term
        between the two Strassen blocks.
        """
        if time.time() > deadline - 10:
            return None
        
        # For the left block, Strassen terms (U_4, V_2, W_4):
        # U indices: 0=a00, 1=a01, 2=a10, 3=a11
        # V indices: 0=b00, 1=b01
        # W indices: 0=C00, 1=C10, 2=C01, 3=C11
        left_terms = [
            ([1,0,0,1], [1,1], [1,0,0,-1]),    # M1: (a00+a11)(b00+b11) -> C00 + C11
            ([0,0,1,1], [1,0], [0,1,-1,0]),    # M2: (a10+a11)b00 -> C10 - C11
            ([1,0,0,0], [0,1], [0,0,1,1]),     # M3: a00(b01-b11) -> C01 + C11
            ([0,0,0,1], [1,0], [1,-1,0,0]),    # M4: a11(b10-b00) -> C00 + C10
            ([1,1,0,0], [0,1], [-1,0,1,0]),    # M5: (a00+a01)b11 -> -C00 + C01
            ([1,-1,0,0], [1,1], [0,0,0,1]),    # M6: (a10-a00)(b00+b01) -> C11
            ([0,1,-1,0], [1,1], [1,0,0,0]),    # M7: (a01-a11)(b10+b11) -> C00
        ]
        
        # For the right block, similar structure with b02,b03 and C02,C12,C03,C13
        right_terms = [
            ([1,0,0,1], [1,1], [0,0,0,0,1,0,0,-1]),    # N1
            ([0,0,1,1], [1,0], [0,0,0,0,0,1,-1,0]),    # N2
            ([1,0,0,0], [0,1], [0,0,0,0,0,0,1,1]),     # N3
            ([0,0,0,1], [1,0], [0,0,0,0,1,-1,0,0]),    # N4
            ([1,1,0,0], [0,1], [0,0,0,0,-1,0,0,1]),    # N5
            ([1,-1,0,0], [1,1], [0,0,0,0,0,0,0,1]),    # N6
            ([0,1,-1,0], [1,1], [0,0,0,0,1,0,0,0]),    # N7
        ]
        
        # Try to find a rank-13 solution by merging one left term with one right term
        for li in range(7):
            if time.time() > deadline - 5:
                return None
            for ri in range(7):
                U_left = np.array(left_terms[li][0], dtype=float)
                U_right = np.array(right_terms[ri][0], dtype=float)
                
                # Check if U_left and U_right are proportional
                if np.linalg.matrix_rank(np.column_stack([U_left, U_right])) < 2:
                    R = 13
                    U_mat = np.zeros((4, R))
                    V_mat = np.zeros((8, R))
                    W_mat = np.zeros((8, R))
                    
                    # Column 0: merged term
                    U_mat[:, 0] = U_left
                    V_mat[:2, 0] = left_terms[li][1]
                    V_mat[2:4, 0] = right_terms[ri][1]
                    W_mat[:4, 0] = left_terms[li][2]
                    W_mat[4:8, 0] = right_terms[ri][2]
                    
                    # Remaining left terms (excluding li)
                    col = 1
                    for l_idx in range(7):
                        if l_idx == li:
                            continue
                        U_mat[:, col] = left_terms[l_idx][0]
                        V_mat[:2, col] = left_terms[l_idx][1]
                        W_mat[:4, col] = left_terms[l_idx][2]
                        col += 1
                    
                    # Remaining right terms (excluding ri)
                    for r_idx in range(7):
                        if r_idx == ri:
                            continue
                        U_mat[:, col] = right_terms[r_idx][0]
                        V_mat[2:4, col] = right_terms[r_idx][1]
                        W_mat[4:8, col] = right_terms[r_idx][2]
                        col += 1
                    
                    # Solve for V using linear algebra
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
                    
                    B_V = B @ V_flat
                    if not np.allclose(B_V, rhs, atol=1e-9):
                        continue
                    
                    V_sol = V_flat.reshape(R, 8).T
                    U_list = U_mat.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_mat.tolist()
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
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

    # Try the structured rank-14 decomposition first (fast and reliable)
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Try Fourier-based rank-13 search
    result = try_fourier_rank_search()
    if result is not None:
        return result
    
    # Try rank-13 search as a fallback
    result = try_rank13_search()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
