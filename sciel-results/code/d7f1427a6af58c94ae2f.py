import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _strassen_2x2_coeffs():
    """Return Strassen coefficients for 2x2 x 2x2."""
    alpha = [
        [1, 0, 0, 1],   # s=0: A00 + A11
        [0, 0, 1, 1],   # s=1: A10 + A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00 + A01
        [-1, 0, 1, 0],  # s=5: A10 - A00
        [0, 1, 0, -1]   # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],   # s=0: B00 + B11
        [1, 0, 0, 0],   # s=1: B00
        [0, 1, 0, -1],  # s=2: B01 - B11
        [-1, 0, 1, 0],  # s=3: B10 - B00
        [0, 0, 0, 1],   # s=4: B11
        [1, 1, 0, 0],   # s=5: B00 + B01
        [0, 0, 1, 1]    # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    return alpha, beta, recon

def _build_44_rank():
    """
    Build a 44-rank decomposition for 4x4 * 4x6.
    
    Strategy:
    - Split B into B1 (4x4, first 4 cols) and B2 (4x2, last 2 cols).
    - C1 = A * B1 (4x4) using direct Strassen on 4x4 matrices (rank 28).
    - C2 = A * B2 (4x2) using schoolbook (rank 16).
    - Total rank: 28 + 16 = 44.
    
    For the 4x4x4 Strassen part:
    - 4x4 matrix A is split into four 2x2 blocks: A00, A01, A10, A11.
    - 4x4 matrix B1 is split into four 2x2 blocks: B100, B101, B110, B111.
    - Use Strassen's 7 multiplications, each of which is a 2x2 x 2x2 multiplication.
    - Each 2x2 x 2x2 multiplication uses rank 4 (schoolbook) or rank 7 (Strassen).
    - If we use schoolbook for the inner 2x2x2, total rank = 7 * 4 = 28.
    - If we use Strassen for the inner 2x2x2, total rank = 7 * 7 = 49 (worse).
    - So we use schoolbook for inner: rank 28.
    """
    n, m, p = 4, 4, 6
    R = 44
    U = np.zeros((n * m, R))  # (16, 44)
    V = np.zeros((m * p, R))  # (24, 44)
    W = np.zeros((p * n, R))  # (24, 44)
    
    alpha, beta, recon = _strassen_2x2_coeffs()
    
    # Part 1: 4x4x4 Strassen for C1 = A * B1 (first 4 columns of B)
    # Rank contribution: 7 * 4 = 28
    col = 0
    for s in range(7):
        # For each Strassen term s:
        # X_s = combination of A's 2x2 blocks
        # Y_s = combination of B1's 2x2 blocks
        # X_s * Y_s is a 2x2 matrix, computed with schoolbook (4 terms)
        
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,u]  (wait, let me be careful)
                # Actually, schoolbook for 2x2 * 2x2:
                # Z[i,j] = sum_k X[i,k] * Y[k,j]
                # So for each output (i,j), we have 2 terms.
                # Total 4 terms for 2x2 output.
                
                # Let's use indices: for output position (out_r, out_c) in the 2x2 result
                # and inner sum index k:
                # Z[out_r, out_c] += X[out_r, k] * Y[k, out_c]
                
                # So we iterate over out_r, out_c, k:
                out_r = u
                out_c = v
                k = 0  # We need to iterate k too... 
                
                # Wait, I need to restructure. For schoolbook 2x2*2x2, 
                # there are 4 independent products if we use the 4 terms directly?
                # No, schoolbook is 4 multiplications for 2x2:
                # Z00 = X00*Y00 + X01*Y10
                # Z01 = X00*Y01 + X01*Y11
                # Z10 = X10*Y00 + X11*Y10
                # Z11 = X10*Y01 + X11*Y11
                # That's 8 multiplications. But Strassen reduces to 7.
                # For rank 28, we want 7 * 4 = 28, so each Strassen term 
                # contributes 4 rank-1 terms.
                # But each Strassen term X_s * Y_s is a 2x2 matrix product.
                # The rank of a 2x2 matrix product tensor is 4 (schoolbook) 
                # or 7 (Strassen). If we use schoolbook for the inner product,
                # each of the 7 outer Strassen terms expands to 4 inner terms.
                # So total = 7 * 4 = 28. Correct.
                
                # So for each outer Strassen term s, and for each of the 4 
                # schoolbook terms (i, j, k) where i,j,k in {0,1} but actually
                # it's 4 terms: (0,0), (0,1), (1,0), (1,1) for the product 
                # X[i,k]*Y[k,j]... no, that's 8.
                
                # Let me reconsider. The schoolbook rank-4 decomposition of 
                # 2x2*2x2 is not standard. Standard schoolbook is rank 8.
                # Strassen is rank 7. The minimal rank of 2x2*2x2 is 7.
                # So I cannot get rank 4 for 2x2*2x2.
                
                # My strategy is flawed. Let me think again.
                
                pass  # This approach doesn't work as described.
                
    # Let me use a different approach:
    # Use the 77-rank Strassen but see if we can reduce it.
    
    # Actually, let's just use the known 77-rank decomposition which is verified.
    # The 44-rank idea doesn't work because 2x2*2x2 has rank 7, not 4.
    
    # Let's try a different structural approach:
    # Use Strassen for the full 4x4*4x4 part with rank 49 (7*7),
    # and schoolbook for 4x4*4x2 with rank 32 (4*4*2).
    # Total: 49 + 32 = 81. Worse than 77.
    
    # Or: schoolbook for 4x4*4x4 (rank 64) + schoolbook for 4x4*4x2 (rank 32) = 96.
    
    # The 77-rank is actually quite good. Let me just return it.
    return None  # Signal to use the 77-rank

def _build_77_rank():
    """Build the 77-rank two-level Strassen decomposition for 4x4 x 4x6."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    alpha, beta, recon = _strassen_2x2_coeffs()
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # U: combine X_s via inner_alpha[si]
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += ca * cr
            
            # V: combine Y_s via inner_beta[si]
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += cb * cc
            
            # W: output Z[i,j] for j in {0,1}
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += cr
                
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += cc
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, R, num_trials=5):
    """Fast vectorized verification of the tensor decomposition."""
    for trial in range(num_trials):
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        C_recon = np.zeros((n, p), dtype=float)
        for i in range(n):
            for r in range(p):
                for k in range(m):
                    C_recon[i, r] += np.sum(U[i*m+k, :] * V[k*p+r, :] * W[r*n+i, :])
        
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
    return True

def _build_schoolbook():
    """Fallback: schoolbook rank-96 decomposition."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for k in range(m):
            for r in range(p):
                U[i * m + k, idx] = 1
                V[k * p + r, idx] = 1
                W[r * n + i, idx] = 1
                idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try 77-rank Strassen decomposition
    try:
        U, V, W = _build_77_rank()
        R = U.shape[1]
        
        # Quick verification
        if time.time() - START_TIME < TIME_LIMIT - 20:
            if _verify_fast(U, V, W, n, m, p, R, num_trials=3):
                def to_list(mat):
                    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat.tolist()]
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook()
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat.tolist()]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
