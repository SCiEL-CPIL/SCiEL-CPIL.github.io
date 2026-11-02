import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-10):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Build decomposition tensor
    decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        decomp += np.einsum('a,b,c->abc', u, v, w)
    
    return np.allclose(decomp, target, atol=tol)


def _strassen_28():
    """Return the rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_forms = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    B_forms = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 5: 1}
    ]

    for block in range(4):
        k0 = block * 2
        
        for s in range(7):
            r = block * 7 + s
            
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}


def _find_rank_27_via_linear_combination():
    """
    Try to find a rank-27 decomposition by checking if the 28 Strassen terms
    are linearly dependent in the tensor space.
    """
    n, m, p = DIMENSIONS
    
    # Get the rank-28 solution
    sol28 = _strassen_28()
    U28 = np.array(sol28['U'], dtype=float)
    V28 = np.array(sol28['V'], dtype=float)
    W28 = np.array(sol28['W'], dtype=float)
    
    nm, mp, pn = n * m, m * p, p * n
    tensor_dim = nm * mp * pn
    
    # Build the matrix where each column is the flattened term
    M = np.zeros((tensor_dim, 28))
    for r in range(28):
        term = np.einsum('a,b,c->abc', U28[:, r], V28[:, r], W28[:, r])
        M[:, r] = term.flatten()
    
    # Check the rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < 28:
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        if S[-1] < 1e-10:
            null_vec = Vt[-1]
            r_remove = np.argmax(np.abs(null_vec))
            U_red = np.delete(U28, r_remove, axis=1)
            V_red = np.delete(V28, r_remove, axis=1)
            W_red = np.delete(W28, r_remove, axis=1)
            
            if _verify_decomposition(U_red, V_red, W_red):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Get the rank-28 solution
    sol28 = _strassen_28()
    
    # Verify it's correct
    if not _verify_decomposition(sol28['U'], sol28['V'], sol28['W']):
        return sol28
    
    # Try to find a rank-27 solution via linear combination
    if time.time() < deadline - 10:
        sol27 = _find_rank_27_via_linear_combination()
        if sol27 is not None and _verify_decomposition(sol27['U'], sol27['V'], sol27['W']):
            return sol27
    
    return sol28

# EVOLVE-BLOCK-END
