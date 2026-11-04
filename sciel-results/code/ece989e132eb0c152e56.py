"""Matrix multiplication 4x4 by 4x8 rank-minimal search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def _build_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _schoolbook():
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

def _verify_float(U, V, W, T, tol=1e-9):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    return np.allclose(T_rec, T, atol=tol)

def _verify_exact(U, V, W, T):
    rank = len(U[0]) if U else 0
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i][c]) * Fraction(V[j][c]) * Fraction(W[k][c])
                if s != Fraction(T[i, j, k]):
                    return False
    return True

def _strassen_2x2x2():
    U = [[1, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 0, -1, 0],
         [1, 1, 0, 1, 0, 0, -1]]
    V = [[1, 1, 0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, -1, 0, 0, 1],
         [1, 0, -1, 0, 1, 0, 1]]
    W = [[1, 0, 0, 1, -1, 0, 1],
         [0, 0, 1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0, 0, 0],
         [1, -1, 1, 0, 0, 1, 0]]
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    n1 = len(U1)
    n2 = len(U2)
    m1 = len(V1)
    m2 = len(V2)
    p1 = len(W1)
    p2 = len(W2)
    U = [[0] * r for _ in range(n1 * n2)]
    V = [[0] * r for _ in range(m1 * m2)]
    W = [[0] * r for _ in range(p1 * p2)]
    for i1 in range(n1):
        for i2 in range(n2):
            row = i1 * n2 + i2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    U[row][col] = U1[i1][c1] * U2[i2][c2]
    for j1 in range(m1):
        for j2 in range(m2):
            row = j1 * m2 + j2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    V[row][col] = V1[j1][c1] * V2[j2][c2]
    for k1 in range(p1):
        for k2 in range(p2):
            row = k1 * p2 + k2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    W[row][col] = W1[k1][c1] * W2[k2][c2]
    return U, V, W

def _strassen_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    return _kron_decomp(U2, V2, W2, U2, V2, W2)

def _direct_sum_4x4x8(U44, V44, W44):
    """Direct sum of two 4x4x4 decompositions for 4x4x8."""
    r = len(U44[0])
    U = []
    for row in U44:
        U.append(row + [0] * r)
    for row in U44:
        U.append([0] * r + row)
    V = []
    for row in V44:
        V.append(row + [0] * r)
    for row in V44:
        V.append([0] * r + row)
    W = []
    for row in W44:
        W.append(row + [0] * r)
    for row in W44:
        W.append([0] * r + row)
    return U, V, W

def _strassen_4x4x8():
    """Construct 4x4x8 via direct sum of two 4x4x4 Strassen decomp."""
    U44, V44, W44 = _strassen_4x4x4()
    return _direct_sum_4x4x8(U44, V44, W44)

def _try_find_low_rank_decomposition(T, target_rank, max_time=60.0):
    """
    Try to find a rank-`target_rank` decomposition using structured random search.
    Uses the fact that for small target ranks, we can try to find solutions where
    many entries are 0, ±1, or ±1/2.
    """
    start = time.time()
    np.random.seed(12345)
    
    n_dim = n * m  # 16
    m_dim = m * p  # 32
    p_dim = p * n  # 32
    
    # Try different structured initializations
    for trial in range(50):
        if time.time() - start > max_time:
            break
        
        # Initialize with sparse random patterns
        U = np.zeros((n_dim, target_rank))
        V = np.zeros((m_dim, target_rank))
        W = np.zeros((p_dim, target_rank))
        
        # Randomly assign non-zero entries from {-1, 1, -0.5, 0.5}
        for c in range(target_rank):
            # Each column has a sparse pattern
            nnz_u = np.random.randint(2, 6)
            nnz_v = np.random.randint(2, 6)
            nnz_w = np.random.randint(2, 6)
            
            idx_u = np.random.choice(n_dim, nnz_u, replace=False)
            idx_v = np.random.choice(m_dim, nnz_v, replace=False)
            idx_w = np.random.choice(p_dim, nnz_w, replace=False)
            
            U[idx_u, c] = np.random.choice([-1, 1, -0.5, 0.5], nnz_u)
            V[idx_v, c] = np.random.choice([-1, 1, -0.5, 0.5], nnz_v)
            W[idx_w, c] = np.random.choice([-1, 1, -0.5, 0.5], nnz_w)
        
        # Alternating minimization to refine
        for it in range(20):
            if time.time() - start > max_time:
                break
            
            # Update U
            for i in range(n_dim):
                M = np.zeros((m_dim * p_dim, target_rank))
                for j in range(m_dim):
                    for k in range(p_dim):
                        M[j * p_dim + k, :] = V[j, :] * W[k, :]
                b = T[i, :, :].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    U[i, :] = sol
                except:
                    pass
            
            # Update V
            for j in range(m_dim):
                M = np.zeros((n_dim * p_dim, target_rank))
                for i in range(n_dim):
                    for k in range(p_dim):
                        M[i * p_dim + k, :] = U[i, :] * W[k, :]
                b = T[:, j, :].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    V[j, :] = sol
                except:
                    pass
            
            # Update W
            for k in range(p_dim):
                M = np.zeros((n_dim * m_dim, target_rank))
                for i in range(n_dim):
                    for j in range(m_dim):
                        M[i * m_dim + j, :] = U[i, :] * V[j, :]
                b = T[:, :, k].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    W[k, :] = sol
                except:
                    pass
            
            # Check convergence
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(T_rec, T, atol=1e-8):
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_rec2 = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                if np.allclose(T_rec2, T, atol=1e-10):
                    # Verify exact with Fractions
                    U_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in U_r.tolist()]
                    V_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in V_r.tolist()]
                    W_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in W_r.tolist()]
                    if _verify_exact(U_frac, V_frac, W_frac, T):
                        return U_frac, V_frac, W_frac
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    T = _build_tensor()
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-based direct sum construction
    try:
        U, V, W = _strassen_4x4x8()
        if _verify_float(U, V, W, T):
            U_frac = [[Fraction(x) for x in row] for row in U]
            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]
            if _verify_exact(U_frac, V_frac, W_frac, T):
                best_rank = len(U_frac[0])
                best = {"U": U_frac, "V": V_frac, "W": W_frac}
                print(f"Strassen 4x4x8: rank {best_rank}")
    except Exception as e:
        print(f"Strassen failed: {e}")
    
    # Try to find lower rank decompositions
    if time.time() < deadline - 30:
        # Try ranks below the Strassen bound
        for target_rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
            if time.time() > deadline - 20:
                break
            try:
                result = _try_find_low_rank_decomposition(T, target_rank, max_time=15.0)
                if result:
                    U_a, V_a, W_a = result
                    if len(U_a[0]) < best_rank:
                        best_rank = len(U_a[0])
                        best = {"U": U_a, "V": V_a, "W": W_a}
                        print(f"Found rank {best_rank}")
                        break
            except Exception as e:
                pass
    
    return best
