import numpy as np
import time

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 130.0


def _now_ok():
    return time.time() < _DEADLINE


def _verify_decomposition(n, m, p, U, V, W, rank, num_tests=10, seed=12345):
    """Vectorized verification of tensor decomposition."""
    U_arr = np.array(U, dtype=np.float64).T
    V_arr = np.array(V, dtype=np.float64).T
    W_arr = np.array(W, dtype=np.float64).T
    
    rng = np.random.default_rng(seed)
    for _ in range(num_tests):
        A = rng.integers(-3, 4, size=(n, m)).astype(np.float64)
        B = rng.integers(-3, 4, size=(m, p)).astype(np.float64)
        C_true = A @ B
        
        C = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U_arr[:, i*m+j], V_arr[:, j*p+k] * W_arr[:, k*n+i])
                C[i, k] = s
        
        if not np.allclose(C, C_true, atol=1e-6):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    # Use the block Strassen decomposition with rank 50
    result = _block_strassen(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, result['U'], result['V'], result['W'], 50):
        # Fall back to schoolbook if verification fails
        return _schoolbook(n, m, p)
    
    return result


def _block_strassen(n, m, p):
    """Block Strassen decomposition for 2x4x7 matrix multiplication."""
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen222(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00][col] += 1
        col += 1
    
    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00][col] += 1
        col += 1
        
        # P2 = b*g
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00][col] += 1
        col += 1
        
        # P3 = c*e
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10][col] += 1
        col += 1
        
        # P4 = d*g
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10][col] += 1
        col += 1
    
    # Build the decomposition
    for j_start in (0, 2):
        for k_start in (0, 2, 4):
            add_strassen222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
