"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
  1. Build a rank-77 decomposition using top-level Strassen (7 terms) where each
     term is a (2x2) x (2x3) product. Each such product uses inner Strassen for
     the 2x2x2 part (7 terms) plus schoolbook for the 2x2x1 part (4 terms) = 11.
     Total rank: 7 * 11 = 77.
  2. Try to reduce rank via structured coordinate descent / column-merging:
     - Attempt to merge pairs of columns that are linearly dependent or near-
       dependent in the U/V/W factor space.
     - Use alternating least squares to refine a smaller-rank candidate.
  3. Fallback to the rank-77 construction if no improvement is found.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: A00 + A11
    {0: 0, 1: 0, 2: 1, 3: 1},   # M2: A10 + A11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00 + A01
    {0: -1, 1: 0, 2: 1, 3: 0},  # M6: A10 - A00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M7: A01 - A11
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: B00 + B11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M3: B01 - B11
    {0: -1, 1: 0, 2: 1, 3: 0},  # M4: B10 - B00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M6: B00 + B01
    {0: 0, 1: 0, 2: 1, 3: 1},   # M7: B10 + B11
]
recon = [
    [(0, 0, 1), (1, 1, 1)],       # M1 -> C00(+1), C11(+1)
    [(1, 0, 1), (1, 1, -1)],      # M2 -> C10(+1), C11(-1)
    [(0, 1, 1), (1, 1, 1)],       # M3 -> C01(+1), C11(+1)
    [(0, 0, 1), (1, 0, 1)],       # M4 -> C00(+1), C10(+1)
    [(0, 0, -1), (0, 1, 1)],      # M5 -> C00(-1), C01(+1)
    [(1, 1, 1)],                  # M6 -> C11(+1)
    [(0, 0, 1)],                  # M7 -> C00(+1)
]


def _build_rank77():
    """Build the rank-77 decomposition using top-level Strassen with 2x2x2+2x2x1 inner."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    # A: 4x4, B: 4x6, C: 4x6
    # Top-level Strassen: split A into 2x2 blocks (4 blocks), B into 2x3 blocks (4 blocks)
    # Wait, B is 4x6. Split into 2x3 blocks? No, that's not right for Strassen.
    # Let's use the structure from inspiration 1:
    # Top-level: A is 4x4, B is 4x6.
    # Split A into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # Split B into 2x3 blocks? No, B is 4x6.
    # Actually, the inspiration uses:
    # A blocks: 4 blocks of 2x2
    # B blocks: 4 blocks of 2x3
    # But B is 4x6, so 2 blocks of 2x6? Or 4 blocks of 2x3?
    # Let's look at the indices: B is m x p = 4 x 6.
    # If we split rows into 2x2 and cols into 2x3, we get 2x2 blocks of 2x3? No.
    # Rows: 0-1, 2-3. Cols: 0-2, 3-5.
    # So B has 4 blocks: B00 (2x3), B01 (2x3), B10 (2x3), B11 (2x3).
    # A has 4 blocks: A00 (2x2), A01 (2x2), A10 (2x2), A11 (2x2).
    # C = A * B is 4x6.
    # C blocks: C00 (2x3), C01 (2x3), C10 (2x3), C11 (2x3).
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Top-level Strassen applies to the 2x2 block matrix multiplication.
    # But the "scalar" multiplications are now (2x2) x (2x3) products.
    # Each (2x2) x (2x3) product has rank 11 (7 Strassen + 4 schoolbook).
    # Total rank: 7 * 11 = 77.
    
    R = 77
    U = np.zeros((n * m, R))  # 16 x 77
    V = np.zeros((m * p, R))  # 24 x 77
    W = np.zeros((p * n, R))  # 24 x 77
    
    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix (combination of A blocks)
        # Y_s is a 2x3 matrix (combination of B blocks)
        # Compute X_s * Y_s (2x3 result)
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # Inner Strassen term si
            # U part: X_s entries with inner_alpha[si] coefficients
            for a in range(4):
                if alpha[si].get(a, 0) != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s].get(r, 0) != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * m + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part: Y_s entries with inner_beta[si] coefficients
            for b in range(4):
                if beta[si].get(b, 0) != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s].get(c, 0) != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * p + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part: output of inner Strassen
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * n + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                for r in range(4):
                    if alpha[s].get(r, 0) != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * m + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s].get(c, 0) != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * p + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * n + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    assert col == R, f"Column mismatch: {col} != {R}"
    return U, V, W


def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _err_of(U, V, W, T):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((T - recon) ** 2)


def _check(U, V, W, T, atol=1e-9):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=atol)


def _try_reduce_rank(U, V, W, T, target_rank, deadline):
    """Try to find a valid decomposition with target_rank columns."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Strategy: randomly select target_rank columns to keep, then refine
    # But better: try to merge columns that are similar
    # For now, use random truncation + refinement
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Randomly select target_rank columns
        cols = np.random.choice(R, size=target_rank, replace=False)
        cols = np.sort(cols)
        
        U_r = U[:, cols].copy()
        V_r = V[:, cols].copy()
        W_r = W[:, cols].copy()
        
        # Add small perturbation
        U_r += np.random.randn(n * m, target_rank) * 0.05
        V_r += np.random.randn(m * p, target_rank) * 0.05
        W_r += np.random.randn(p * n, target_rank) * 0.05
        
        err = _err_of(U_r, V_r, W_r, T)
        
        # Alternating minimization
        for it in range(300):
            if time.time() > deadline - 5:
                break
            if err < 1e-12:
                break
            
            recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            diff = T - recon
            
            grad_U = -2.0 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
            grad_V = -2.0 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
            grad_W = -2.0 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
            
            lr = 0.01
            U_new = U_r - lr * grad_U
            V_new = V_r - lr * grad_V
            W_new = W_r - lr * grad_W
            
            new_err = _err_of(U_new, V_new, W_new, T)
            if new_err < err:
                U_r, V_r, W_r = U_new, V_new, W_new
                err = new_err
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W, True
    return None, None, None, False


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 115  # safety margin
    
    T = _build_target_tensor()
    
    # Build rank-77 decomposition
    U77, V77, W77 = _build_rank77()
    
    # Verify
    assert _check(U77, V77, W77, T, atol=1e-6), "Rank-77 verification failed"
    
    best_rank = 77
    best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()
    
    # Try to reduce rank
    for target_rank in range(76, 50, -1):
        if time.time() > deadline - 20:
            break
        
        U_new, V_new, W_new, success = _try_reduce_rank(U77, V77, W77, T, target_rank, deadline)
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            break
    
    # Return the best result
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
