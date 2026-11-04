import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # 7 Strassen terms
        # M1 = (X11+X22)(Y11+Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M2 = (X21+X22)Y11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        # M3 = X11(Y12-Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M4 = X22(Y21-Y11)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        # M5 = (X11+X12)Y22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        # M6 = (X21-X11)(Y11+Y12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M7 = (X12-X22)(Y21+Y22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    return U, V, W

def _als_decomposition_vectorized(T, n, m, p, rank, max_iter=100, tol=1e-8):
    """Vectorized ALS decomposition of tensor T into rank-r sum of outer products."""
    np.random.seed(42)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    # Precompute index arrays
    i_idx = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    j_idx = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    k_idx = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    for it in range(max_iter):
        # Update U
        for r in range(rank):
            Vr = V[:, r]
            Wr = W[:, r]
            # num[i*m+j] = sum_{j,k} T[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
            num = np.zeros(n*m)
            den = np.zeros(n*m)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        num[i*m+j] += T[i,j,k] * Vr[j*p+k] * Wr[k*n+i]
                        den[i*m+j] += Vr[j*p+k]**2 * Wr[k*n+i]**2
            mask = den > 1e-12
            U[mask, r] = num[mask] / den[mask]
        
        # Update V
        for r in range(rank):
            Ur = U[:, r]
            Wr = W[:, r]
            num = np.zeros(m*p)
            den = np.zeros(m*p)
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        num[j*p+k] += T[i,j,k] * Ur[i*m+j] * Wr[k*n+i]
                        den[j*p+k] += Ur[i*m+j]**2 * Wr[k*n+i]**2
            mask = den > 1e-12
            V[mask, r] = num[mask] / den[mask]
        
        # Update W
        for r in range(rank):
            Ur = U[:, r]
            Vr = V[:, r]
            num = np.zeros(p*n)
            den = np.zeros(p*n)
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        num[k*n+i] += T[i,j,k] * Ur[i*m+j] * Vr[j*p+k]
                        den[k*n+i] += Ur[i*m+j]**2 * Vr[j*p+k]**2
            mask = den > 1e-12
            W[mask, r] = num[mask] / den[mask]
        
        # Check convergence
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
        if np.max(np.abs(T_recon - T)) < tol:
            break
    
    return U, V, W

def _try_merge_terms(U, V, W, n, m, p, max_tries=1000, time_budget=5):
    """Try to find two terms that can be merged or cancelled to reduce rank."""
    start_time = time.time()
    rank = U.shape[1]
    best_rank = rank
    
    # Try removing pairs of terms and see if we can compensate
    for i in range(rank):
        if time.time() - start_time > time_budget:
            break
        for j in range(i+1, rank):
            if time.time() - start_time > time_budget:
                break
            # Create a copy without terms i and j
            keep = [k for k in range(rank) if k != i and k != j]
            U_test = U[:, keep].copy()
            V_test = V[:, keep].copy()
            W_test = W[:, keep].copy()
            
            # Check if the remaining terms still give a valid decomposition
            # (This is a quick check - in reality we'd need to verify)
            # For now, just track if we found a valid lower rank
            # This is a placeholder - actual merging is complex
    
    return None

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44, V44, W44
    best_rank = 44
    
    # Verify the base decomposition
    assert _verify_decomposition(U44, V44, W44, n, m, p, tol=1e-6), "Base decomposition failed!"
    
    # Try to find lower-rank decompositions using ALS
    # Be conservative with time budget
    for r in [43, 42, 41, 40, 39, 38]:
        if time.time() > deadline - 30:
            break
        
        # Try a few ALS initializations
        found = False
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(42 + trial)
            U, V, W = _als_decomposition_vectorized(T, n, m, p, r, max_iter=50, tol=1e-6)
            
            # Check if we get a valid decomposition
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                # Round to half-integers if close
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-4):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    found = True
                    break
                else:
                    # Use unrounded if valid
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
