"""Improved program: top-level Strassen with inner Strassen+schoolbook for 4x4*4x6."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Top-level Strassen (7 terms) where each term is a (2x2) x (2x3) product.
    For each (2x2)x(2x3): use inner Strassen for the 2x2x2 part (7 mults) + 
    schoolbook for the 2x2x1 part (4 mults) = 11 mults per term.
    Total rank: 7 * 11 = 77.
    Then attempt rank reduction by checking for linear dependencies.
    """
    t0 = time.time()
    deadline = t0 + 115  # safety margin before 120s limit
    
    n, m, p = 4, 4, 6
    
    # Top-level Strassen coefficients
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},   # M1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0},  # M5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}   # M6: A01 - A11
    ]
    
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M1: B00
        {0: 0, 1: 1, 2: 0, 3: -1},  # M2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0},  # M3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}    # M6: B10 + B11
    ]
    
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # M0 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],    # M1 -> C10, -C11
        [(0, 1, 1), (1, 1, 1)],     # M2 -> C01, C11
        [(0, 0, 1), (1, 0, 1)],     # M3 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],    # M4 -> -C00, C01
        [(1, 1, 1)],                # M5 -> C11
        [(0, 0, 1)]                 # M6 -> C00
    ]
    
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                coeff_a = inner_alpha[si].get(a, 0)
                if coeff_a != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        coeff_r = alpha[s].get(r, 0)
                        if coeff_r != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += coeff_a * coeff_r
            
            for b in range(4):
                coeff_b = inner_beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b_idx * 3 + zb) * 4 + (a_idx * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                
                for a_idx, b_idx, gamma in recon[s]:
                    w_idx = (b_idx * 3 + 2) * 4 + (a_idx * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Attempt rank reduction: check for linearly dependent columns
    if time.time() < deadline - 10:
        # Build the full tensor T[i,j,k] = sum_m U[i,m] V[j,m] W[k,m]
        # Vectorize: T_vec = sum_m (U[:,m] ⊗ V[:,m] ⊗ W[:,m])
        # If the 77 vectors are linearly dependent, we can reduce rank.
        # Check by forming matrix M where each column is U[:,m] ⊗ V[:,m] ⊗ W[:,m]
        # But 16*24*24 = 9216 rows, 77 cols. Check rank.
        M = np.zeros((16*24*24, R))
        for m_idx in range(R):
            M[:, m_idx] = np.outer(np.outer(U[:, m_idx], V[:, m_idx]), W[:, m_idx]).flatten()
        
        # Compute rank
        try:
            rank_M = np.linalg.matrix_rank(M, tol=1e-10)
            if rank_M < R:
                # Try to find a minimal subset of columns that spans
                # Use QR with column pivoting
                Q, Rmat, piv = np.linalg.qr(M.T, mode='complete')
                # Actually, let's use a simpler approach:
                # Find which columns are linearly independent
                # Use SVD
                U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
                # rank is number of significant singular values
                tol_s = max(M.shape) * np.finfo(float).eps * s_vals[0]
                effective_rank = np.sum(s_vals > tol_s)
                if effective_rank < R:
                    # Keep only the first effective_rank columns (after pivoting)
                    # This is approximate; for exact, we'd need rational arithmetic
                    # For now, just note the rank
                    pass
        except Exception:
            pass
    
    # Convert to lists of integers/floats
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                elif abs(2*x - round(2*x)) < 1e-9:
                    new_row.append(round(2*x) / 2.0)
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
