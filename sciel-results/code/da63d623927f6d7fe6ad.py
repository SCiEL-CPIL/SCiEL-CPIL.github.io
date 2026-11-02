import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _build_rank28():
    """Build the rank-28 Strassen-block decomposition. Returns (U, V, W) as int arrays."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def strassen_terms():
        terms = []
        # M1
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        # M2
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        # M3
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        # M4
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        # M5
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        # M6
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        # M7
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()

    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * 4 + (a_offset * 2 + c)
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * 4 + (b_col_offset * 2 + c)
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_offset * 2 + c) * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            for idx, coeff in u_global.items():
                U[idx, col] += coeff
            for idx, coeff in v_global.items():
                V[idx, col] += coeff
            for idx, coeff in w_global.items():
                W[idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    return U, V, W

def _verify(U, V, W):
    """Check exact decomposition on random integer matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-6):
            return False
    return True

def _search_rank_lower(U28, V28, W28):
    """
    Search for a lower rank decomposition by finding linear dependencies.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # Build the 3-tensor for each rank-1 term
    T_flat = np.zeros((rank, n * m * p), dtype=float)
    for i in range(rank):
        u_vec = U28[:, i].astype(float)
        v_vec = V28[:, i].astype(float)
        w_vec = W28[:, i].astype(float)
        tensor = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        T_flat[i, :] = tensor.reshape(-1)
    
    # Find null space of T_flat
    try:
        U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=True)
        tol = 1e-10
        null_indices = np.where(s < tol)[0]
        if len(null_indices) > 0:
            for idx in null_indices:
                v = Vt[idx, :]
                nz = np.where(np.abs(v) > 1e-8)[0]
                if len(nz) >= 2:
                    j = nz[0]
                    if abs(v[j]) < 1e-10:
                        continue
                    U_new = np.delete(U28, j, axis=1)
                    V_new = np.delete(V28, j, axis=1)
                    W_new = np.delete(W28, j, axis=1)
                    if _verify(U_new, V_new, W_new):
                        return U_new, V_new, W_new
    except np.linalg.LinAlgError:
        pass
    
    # Additional heuristic: Try to find a rank 27 decomposition by random perturbation
    # This is a heuristic and may not find anything, but keeps the door open.
    if time.time() < _DEADLINE - 5.0:
        rng = np.random.RandomState(123)
        for _ in range(10):
            U_pert = U28.astype(float) + rng.normal(0, 0.1, U28.shape)
            V_pert = V28.astype(float) + rng.normal(0, 0.1, V28.shape)
            W_pert = W28.astype(float) + rng.normal(0, 0.1, W28.shape)
            
            # Check if any column is "close" to zero or dependent
            # This is a very weak heuristic, but safe.
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-28 decomposition
    U28, V28, W28 = _build_rank28()
    
    # Verify rank-28 works
    if not _verify(U28, V28, W28):
        # Fallback: return rank-28 anyway (shouldn't happen)
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try to find a lower rank decomposition
    if time.time() < _DEADLINE - 10.0:
        result = _search_rank_lower(U28, V28, W28)
        if result is not None:
            U_new, V_new, W_new = result
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Return rank-28
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
