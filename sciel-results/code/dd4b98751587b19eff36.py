"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
  1. Build a rank-84 decomposition by splitting the 4x4 * 4x6 product into
     three independent 4x4 * 4x2 subproblems (one per pair of output columns).
     Each 4x4 * 4x2 is computed by 4 independent 2x2 * 2x2 Strassen products
     (rank 7 each), giving 4*7 = 28 per subproblem, total 84.
  2. Use coordinate descent / alternating minimization to try to reduce rank
     by finding a valid decomposition with fewer columns.
  3. Fallback to the rank-84 construction if no improvement is found.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
# A blocks indexed 0..3: 00, 01, 10, 11
# B blocks indexed 0..3: 00, 01, 10, 11
# C blocks indexed 0..3: 00, 01, 10, 11

# alpha[s][a] = coefficient of A-block a in M_s
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: A00 + A11
    {0: 0, 1: 0, 2: 1, 3: 1},   # M2: A10 + A11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00 + A01
    {0: -1, 1: 0, 2: 1, 3: 0},  # M6: A10 - A00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M7: A01 - A11
]
# beta[s][b]  = coefficient of B-block b in M_s
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: B00 + B11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M3: B01 - B11
    {0: -1, 1: 0, 2: 1, 3: 0},  # M4: B10 - B00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M6: B00 + B01
    {0: 0, 1: 0, 2: 1, 3: 1},   # M7: B10 + B11
]
# recon[s] = list of (a_block, b_block, gamma) for C = sum_s recon[s] * M_s
recon = [
    [(0, 0, 1), (1, 1, 1)],       # M1 -> C00(+1), C11(+1)
    [(1, 0, 1), (1, 1, -1)],      # M2 -> C10(+1), C11(-1)
    [(0, 1, 1), (1, 1, 1)],       # M3 -> C01(+1), C11(+1)
    [(0, 0, 1), (1, 0, 1)],       # M4 -> C00(+1), C10(+1)
    [(0, 0, -1), (0, 1, 1)],      # M5 -> C00(-1), C01(+1)
    [(1, 1, 1)],                  # M6 -> C11(+1)
    [(0, 0, 1)],                  # M7 -> C00(+1)
]


def _build_rank84():
    """Build the rank-84 decomposition using 3x (4x4 * 4x2) Strassen blocks."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    # A: 4x4, B: 4x6, C: 4x6
    # Split B and C into 3 pairs of columns: (0,1), (2,3), (4,5)
    # For each pair, we compute a 4x4 * 4x2 product
    # Split 4x4 A into four 2x2 blocks: A00, A01, A10, A11
    # Split 4x2 B_pair into two 2x2 blocks: B0, B1
    # Split 4x2 C_pair into two 2x2 blocks: C0, C1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Actually, for 4x4 * 4x2:
    # A is 4x4, B is 4x2, C is 4x2
    # Split A into 2x2 blocks: A = [A00 A01; A10 A11]
    # Split B into 2x2 blocks: B = [B0; B1] where B0 is 2x2 (rows 0-1), B1 is 2x2 (rows 2-3)
    # C = [C0; C1] where C0 is 2x2, C1 is 2x2
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Each of C0 and C1 is a sum of two 2x2*2x2 products
    # We can use Strassen for each 2x2*2x2 product
    # But we need to combine them...
    
    # Actually, let me think about this differently.
    # For the 4x4 * 4x2 product, we can view it as:
    # C = A * B where A is 4x4, B is 4x2
    # We can split this into 4 independent 2x2 * 2x2 products if we
    # partition the rows of A and B appropriately.
    
    # Let me use a block approach:
    # A = [A00 A01; A10 A11] (each 2x2)
    # B = [B0; B1] (each 2x2, where B0 is top 2 rows, B1 is bottom 2 rows)
    # C = [C0; C1] (each 2x2)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # For C0 = A00*B0 + A01*B1:
    # This is a sum of two 2x2*2x2 products.
    # We can compute each with Strassen (7 terms each), total 14 terms for C0.
    # Similarly 14 terms for C1.
    # Total for 4x4*4x2: 28 terms.
    # For 3 such subproblems: 84 terms total.
    
    rank = 84
    U = np.zeros((n * m, rank), dtype=np.float64)  # 16 x 84
    V = np.zeros((m * p, rank), dtype=np.float64)  # 24 x 84
    W = np.zeros((p * n, rank), dtype=np.float64)  # 24 x 84
    
    col = 0
    # For each of the 3 column pairs (k_pair = 0,1,2 corresponds to cols 2*k_pair, 2*k_pair+1)
    for k_pair in range(3):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        # For C0 (top 2 rows of C, i.e., i=0,1) and C1 (bottom 2 rows, i=2,3)
        for i_block in range(2):  # 0 for top, 1 for bottom
            i0 = 2 * i_block
            i1 = 2 * i_block + 1
            # For each of the two 2x2*2x2 products in this block
            for a_block in range(2):  # 0 for A0x, 1 for A1x
                a0 = 2 * a_block
                a1 = 2 * a_block + 1
                # A block indices: (i0,a0), (i0,a1), (i1,a0), (i1,a1)
                # B block indices: (a0,k0), (a0,k1), (a1,k0), (a1,k1)
                # C block indices: (k0,i0), (k0,i1), (k1,i0), (k1,i1)
                
                # Strassen for this 2x2*2x2 product
                for s in range(7):
                    # A coefficients
                    for a_idx in range(4):
                        coeff_a = alpha[s].get(a_idx, 0)
                        if coeff_a != 0:
                            ai, aj = a_idx // 2, a_idx % 2
                            U[(i0 + ai) * m + (a0 + aj), col] += coeff_a
                    # B coefficients
                    for b_idx in range(4):
                        coeff_b = beta[s].get(b_idx, 0)
                        if coeff_b != 0:
                            bi, bj = b_idx // 2, b_idx % 2
                            V[(a0 + bi) * p + (k0 + bj), col] += coeff_b
                    # C reconstruction
                    for (c_a, c_b, gamma) in recon[s]:
                        ci, cj = c_a, c_b  # ci is row-block of C (0 or 1), cj is col-block (0 or 1)
                        # C index: (k0 + cj, i0 + ci)
                        W[(k0 + cj) * n + (i0 + ci), col] += gamma
                    col += 1
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
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


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 115  # safety margin
    
    T = _build_target_tensor()
    
    # Build rank-84 decomposition
    U84, V84, W84 = _build_rank84()
    
    # Verify
    assert _check(U84, V84, W84, T, atol=1e-6), "Rank-84 verification failed"
    
    best_rank = 84
    best_U, best_V, best_W = U84.copy(), V84.copy(), W84.copy()
    
    # Try to reduce rank via optimization
    # Strategy: start from rank-84, try to find a valid decomposition with fewer columns
    # by randomly initializing and running alternating minimization
    
    for target_rank in range(83, 60, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random restarts
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            # Initialize: take first target_rank columns of the rank-84 solution
            # and add some random perturbation
            U = U84[:, :target_rank].copy()
            V = V84[:, :target_rank].copy()
            W = W84[:, :target_rank].copy()
            
            # Add small random perturbation
            scale = 0.1
            U += np.random.randn(n * m, target_rank) * scale
            V += np.random.randn(m * p, target_rank) * scale
            W += np.random.randn(p * n, target_rank) * scale
            
            err = _err_of(U, V, W, T)
            
            # Alternating minimization
            for it in range(500):
                if time.time() > deadline - 15:
                    break
                if err < 1e-12:
                    break
                
                # Update U
                # For each column r, U[:,r] should minimize sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
                # This is a linear least squares problem for each column
                # Fix V and W, update U
                # For column r: U[:,r] = argmin_U sum_{a,b,c} (T[a,b,c] - U[a,r]*V[b,r]*W[c,r] - sum_{r'!=r} U[a,r']*V[b,r']*W[c,r'])^2
                # This is complex. Let's use a simpler approach: gradient descent.
                
                # Compute gradient of err w.r.t. U
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - recon  # (n*m, m*p, p*n)
                
                # d_err/d_U[a,r] = -2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                # Use einsum: grad_U[a,r] = -2 * einsum('abc,br,cr->ar', diff, V, W)
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Simple gradient step with adaptive learning rate
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                new_err = _err_of(U, V, W, T)
                if new_err < err:
                    err = new_err
                else:
                    # Revert
                    U += lr * grad_U
                    V += lr * grad_V
                    W += lr * grad_W
                    # Try smaller step
                    lr *= 0.5
                    if lr < 1e-6:
                        break
        
        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
    
    # Return the best result
    # Convert to lists for the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
