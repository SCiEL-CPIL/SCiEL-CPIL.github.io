import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Construct rank-28 Strassen decomposition
    rank28 = 28
    U28 = np.zeros((n * m, rank28), dtype=int)
    V28 = np.zeros((m * p, rank28), dtype=int)
    W28 = np.zeros((p * n, rank28), dtype=int)

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]

    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)

    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r

    col = 0
    block_configs = [
        ('L', 'TL', 'L'),
        ('R', 'BL', 'L'),
        ('L', 'TR', 'R'),
        ('R', 'BR', 'R'),
    ]

    for a_side, b_block, c_side in block_configs:
        for u_l, v_l, w_l in strassen_terms:
            for idx, coeff in u_l:
                U28[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_l:
                V28[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_l:
                W28[map_w_c(c_side, idx), col] += coeff
            col += 1

    def verify_decomp(U, V, W, tol=1e-10):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_recon - T)) > tol:
            return False
        return True

    if not verify_decomp(U28, V28, W28):
        rank = 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Optimized solver for W given U, V
    def solve_for_W(U_fixed, V_fixed):
        K = U_fixed.shape[1]
        # Construct M of shape (n*m*m*p, K)
        # M[(a, b), r] = U[a, r] * V[b, r]
        # U_fixed shape: (n*m, K)
        # V_fixed shape: (m*p, K)
        # We need to map indices.
        # a is in [0, n*m), b is in [0, m*p)
        # The tensor index is (a, b, c).
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        
        # Create M
        # M rows correspond to (a, b) pairs. Total rows = (n*m) * (m*p)
        # M cols correspond to r. Total cols = K
        # M[i, r] where i = a * (m*p) + b
        # M[i, r] = U[a, r] * V[b, r]
        
        # Vectorized construction
        # U_expanded: (n*m, 1, K)
        # V_expanded: (1, m*p, K)
        # Product: (n*m, m*p, K)
        # Reshape to (n*m * m*p, K)
        M = (U_fixed[:, np.newaxis, :] * V_fixed[np.newaxis, :, :]).reshape(n * m * m * p, K)
        
        # T_flat shape: (n*m*m*p, p*n)
        # T_flat[i, c] = T[a, b, c] where i = a * (m*p) + b
        # T is (n*m, m*p, p*n)
        # We need to reshape T such that (a, b) are combined into first dim.
        # T.transpose(0, 1, 2) is already (n*m, m*p, p*n)
        # Reshape to (n*m * m*p, p*n)
        T_flat = T.reshape(n * m * m * p, p * n)
        
        # Solve M * W^T = T_flat
        # W^T has shape (K, p*n)
        # M has shape (n*m*m*p, K)
        # This is an overdetermined system if n*m*m*p > K.
        # We want to find W (shape p*n, K)
        # Let Y = W^T (shape K, p*n)
        # M * Y = T_flat
        # Y = lstsq(M, T_flat)
        
        try:
            Y, residuals, rank_M, sv = np.linalg.lstsq(M, T_flat, rcond=None)
            W_solution = Y.T
            return W_solution
        except:
            return None

    found = False
    best_rank = 28
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()
    
    # Phase 1: Drop one term
    for elim in range(28):
        if time.time() > deadline - 5:
            break
        keep = [i for i in range(28) if i != elim]
        U_fixed = U28[:, keep].astype(float)
        V_fixed = V28[:, keep].astype(float)
        
        W_solution = solve_for_W(U_fixed, V_fixed)
        if W_solution is not None:
            if verify_decomp(U_fixed, V_fixed, W_solution):
                W_rounded = np.round(W_solution * 2) / 2
                if verify_decomp(U_fixed, V_fixed, W_rounded):
                    found = True
                    best_U, best_V, best_W = U_fixed, V_fixed, W_rounded
                    best_rank = 27
                    break

    # Phase 2: Perturb
    if not found and time.time() < deadline - 5:
        np.random.seed(42)
        scales = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
        for elim in range(28):
            if time.time() > deadline - 5:
                break
            keep = [i for i in range(28) if i != elim]
            for scale in scales:
                if time.time() > deadline - 5:
                    break
                U_fixed = U28[:, keep].astype(float) + np.random.randn(n * m, 27) * scale
                V_fixed = V28[:, keep].astype(float) + np.random.randn(m * p, 27) * scale
                
                W_solution = solve_for_W(U_fixed, V_fixed)
                if W_solution is not None:
                    if verify_decomp(U_fixed, V_fixed, W_solution):
                        W_rounded = np.round(W_solution * 2) / 2
                        if verify_decomp(U_fixed, V_fixed, W_rounded):
                            found = True
                            best_U, best_V, best_W = U_fixed, V_fixed, W_rounded
                            best_rank = 27
                            break
            if found:
                break

    if best_rank < 28:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
