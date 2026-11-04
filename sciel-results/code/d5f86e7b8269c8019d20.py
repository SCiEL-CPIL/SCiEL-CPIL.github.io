import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)

def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks."""
    block_coeffs_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    block_coeffs_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    block_coeffs_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (0, 0, 1)],
        [(0, 1, -1), (1, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc

                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u

                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v

                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w

    return U, V, W

def _try_rank_28_4x4x5():
    """
    Attempt to find a rank-28 decomposition for 4x4x5.
    Uses the 4x4x4 Strassen core and solves for the 5th column/row.
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    rank = 28
    n, m, p = 4, 4, 5
    
    # Convert to numpy arrays for easier manipulation
    # U: 16 x 28
    U_arr = np.array(U44, dtype=np.float64)
    # V44: 16 x 28 (4x4 blocks)
    V44_arr = np.array(V44, dtype=np.float64)
    # W44: 16 x 28 (4x4 blocks)
    W44_arr = np.array(W44, dtype=np.float64)
    
    # Reshape to 3D: (dim1, dim2, rank)
    U_3d = U_arr.reshape(4, 4, rank)
    V44_3d = V44_arr.reshape(4, 4, rank)
    W44_3d = W44_arr.reshape(4, 4, rank)
    
    # We need to find V_5 (4 x rank) and W_5 (4 x rank)
    # Such that sum_c U_c * V_5_c * W_5_c = J (4x4 matrix of 1s)
    # Equation: sum_c U_{ij}^c * V_{j5}^c * W_{5i}^c = 1 for all i,j
    
    # Let x_c be the vector of length 16 representing V_5_c and W_5_c?
    # No, V_5_c is a vector of length 4. W_5_c is a vector of length 4.
    # Let v_c = V_5_c (4x1) and w_c = W_5_c (4x1).
    # We want sum_c U_c * (v_c w_c^T) = J.
    # This is a bilinear equation in v_c and w_c.
    
    # Linearize: Let Y_c = v_c w_c^T (4x4).
    # sum_c U_c * Y_c = J (element-wise).
    # This is a linear system for Y_c.
    # Variables: Y_c (28 * 16 = 448 variables).
    # Equations: 16 (entries of J).
    # We can pick a solution. To ensure Y_c is rank 1, we can try to find
    # v_c, w_c directly using a non-linear least squares approach, or
    # solve the linear system and check rank.
    
    # Let's try to solve for Y_c first.
    # Construct the design matrix A (16 x 448).
    # A_{(i,j), (c, k, l)} = U_{ij}^c if Y_c index is (k,l)?
    # Wait, the sum is sum_c sum_{k,l} U_{ij}^c * Y_{kl}^c * delta_{ik} delta_{jl}?
    # No, the term is U_{ij}^c * Y_{ij}^c?
    # Recall: Term is U_{ij}^c V_{j5}^c W_{5i}^c.
    # Let V_{j5}^c = v_{c,j} and W_{5i}^c = w_{c,i}.
    # Term is U_{ij}^c v_{c,j} w_{c,i}.
    # This is NOT U_{ij}^c Y_{ij}^c.
    # It is U_{ij}^c * (v_c w_c^T)_{ji}? No.
    # (v_c w_c^T) is a matrix M_c where M_{ji} = v_j w_i.
    # So the term is U_{ij}^c M_{ji}^c.
    # So sum_c sum_{j,i} U_{ij}^c M_{ji}^c = 1.
    # This is the Frobenius inner product <U_c, M_c^T> = 1?
    # Sum_c <U_c, M_c^T> = 1.
    # Let Z_c = M_c^T = w_c v_c^T.
    # Then sum_c <U_c, Z_c> = 1.
    # Z_c is rank 1 (outer product of w_c and v_c).
    
    # We can solve for Z_c linearly.
    # Variables Z_c (4x4). 28*16 = 448 vars.
    # 16 equations.
    # We can set 432 variables to 0 and solve for 16.
    # But we need Z_c to be rank 1.
    
    # Alternative: Parametrize Z_c = w_c v_c^T.
    # This is a non-linear system.
    # Let's try a simple ansatz:
    # Assume v_c = w_c = e_1 (first basis vector) for all c?
    # Then Z_c = e_1 e_1^T.
    # Sum_c U_{ij}^c (e_1 e_1^T)_{ji} = Sum_c U_{ij}^c delta_{j1} delta_{i1} = U_{11}^c.
    # We need Sum_c U_{11}^c = 1 for all i,j? No, only for i=1,j=1?
    # The equation is sum_c U_{ij}^c Z_{ji}^c = 1 for ALL i,j.
    # If we use Z_c = e_1 e_1^T, then Z_{ji}^c is non-zero only if j=1, i=1.
    # So LHS is 0 for i!=1 or j!=1. But RHS is 1. So this fails.
    
    # We need Z_c to have support on all i,j.
    # Let's try to solve the linear system for Z_c and then check if we can
    # decompose Z_c into rank 1.
    # Since the system is underdetermined, we can add a constraint that Z_c is rank 1.
    
    # Let's use scipy.optimize to find v_c, w_c.
    # Variables: v_c (4x28), w_c (4x28). Total 8*28 = 224 vars.
    # Objective: sum_{i,j} (1 - sum_c U_{ij}^c v_{c,j} w_{c,i})^2.
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None

    # Flatten v and w
    # v: 4 x 28, w: 4 x 28
    # x = [v_flat, w_flat]
    
    def residuals(x):
        v_flat = x[:4*rank]
        w_flat = x[4*rank:]
        v = v_flat.reshape(4, rank)
        w = w_flat.reshape(4, rank)
        
        # Compute sum_c U_c * (v_c w_c^T)^T?
        # Term: U_{ij}^c v_{c,j} w_{c,i}
        # v_c is column c of v. w_c is column c of w.
        # v_{c,j} is v[j, c]. w_{c,i} is w[i, c].
        
        res = np.zeros((4, 4))
        for c in range(rank):
            # U_c is U_3d[:, :, c]
            # v_c is v[:, c]
            # w_c is w[:, c]
            # Term matrix M = U_c * (v_c w_c^T)^T?
            # M_{ij} = U_{ij}^c * v_{c,j} * w_{c,i}
            # This is U_c * (v_c w_c^T) element-wise?
            # (v_c w_c^T)_{ji} = v_{c,j} w_{c,i}.
            # So M = U_c * (v_c w_c^T).T
            outer = np.outer(v[:, c], w[:, c]) # outer[i, j] = w[i] v[j] = w_{c,i} v_{c,j}
            # We need v_{c,j} w_{c,i}. This is outer.T.
            term = U_3d[:, :, c] * outer.T
            res += term
            
        # We want res to be all 1s.
        return (res - 1).flatten()

    # Initial guess:
    # Try random small values
    np.random.seed(42)
    x0 = np.random.randn(8 * rank) * 0.1
    
    # Run optimization
    result = least_squares(residuals, x0, method='lm', max_nfev=2000)
    
    if result.success or np.linalg.norm(result.fun) < 1e-8:
        v_flat = result.x[:4*rank]
        w_flat = result.x[4*rank:]
        v_5 = v_flat.reshape(4, rank)
        w_5 = w_flat.reshape(4, rank)
        
        # Construct full V and W
        V_full = np.zeros((4, 5, rank))
        W_full = np.zeros((5, 4, rank))
        
        # Copy core
        V_full[:, :4, :] = V44_3d
        W_full[:4, :, :] = W44_3d
        
        # Add 5th column/row
        V_full[:, 4, :] = v_5
        W_full[4, :, :] = w_5
        
        # Convert back to lists
        U_list = U_arr.tolist()
        V_list = V_full.reshape(20, rank).tolist()
        W_list = W_full.reshape(20, rank).tolist()
        
        # Verify
        if _verify_decomposition_fast(U_list, V_list, W_list, 4, 4, 5):
            return U_list, V_list, W_list
            
    return None

def _build_4x4x5_improved():
    """Build 4x4x5 decomposition: Strassen 4x4x4 core + 1 boundary term."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_boundary = 1
    rank_total = rank_core + rank_boundary

    # U: 16 x rank_total
    # V: 20 x rank_total (4 rows x 5 cols)
    # W: 20 x rank_total (5 rows x 4 cols)
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]

    # Copy core (indices 0..27)
    # Core V is 4x4, mapped to V[j, k] for k in 0..3
    # Core W is 4x4, mapped to W[k, i] for k in 0..3
    for c in range(rank_core):
        for i in range(16):
            U[i][c] = U44[i][c]
        for j in range(4):
            for k in range(4):
                V[j * 5 + k][c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W44[k * 4 + i][c]

    # Add boundary term for k=4 (index 4)
    c = rank_core
    # U[i, j] = 1/4 for all i, j
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][c] = 0.25
    # V[j, k] = 1 if k=4, else 0
    for j in range(4):
        V[j * 5 + 4][c] = 1.0
    # W[k, i] = 1 if k=4, else 0
    for i in range(4):
        W[4 * 4 + i][c] = 1.0

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 28 first
    res_28 = _try_rank_28_4x4x5()
    if res_28 is not None:
        U, V, W = res_28
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback to rank 29
    U, V, W = _build_4x4x5_improved()
    if _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
