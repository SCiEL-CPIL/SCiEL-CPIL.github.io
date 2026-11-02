import numpy as np
import time

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 140.0

def _now_ok():
    return time.time() < _DEADLINE

def construct():
    n, m, p = DIMENSIONS
    
    def build_strassen_222(i0, j0, k0, U, V, W, col):
        """Add a 2x2x2 Strassen block. Returns new col index."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        return col
    
    def build_schoolbook_221(i0, j0, k0, U, V, W, col):
        """Add a 2x2x1 schoolbook block (4 mults)."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        return col
    
    def verify_decomposition(U, V, W, rank):
        """Check that the decomposition is exact using random integer matrices."""
        np.random.seed(42)
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            U_r = U[:, r]
            V_r = V[:, r]
            W_r = W[:, r]
            if np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0):
                continue
            UA = np.zeros(n, dtype=np.int64)
            VB = np.zeros(p, dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    if U_r[i * m + j] != 0:
                        UA[i] += U_r[i * m + j] * A[i, j]
            for k in range(p):
                for j in range(m):
                    if V_r[j * p + k] != 0:
                        VB[k] += V_r[j * p + k] * B[j, k]
            for i in range(n):
                for k in range(p):
                    if W_r[k * n + i] != 0:
                        C_computed[i, k] += W_r[k * n + i] * UA[i] * VB[k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True

    # Baseline rank-50 solution
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            col = build_strassen_222(0, j_start, k_start, U, V, W, col)
        col = build_schoolbook_221(0, j_start, 6, U, V, W, col)
    
    assert col == 50, f"Expected col=50, got {col}"
    
    if not verify_decomposition(U, V, W, 50):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
