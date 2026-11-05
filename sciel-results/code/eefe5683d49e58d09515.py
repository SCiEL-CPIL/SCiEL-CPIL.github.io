"""Combined: structured rank-44 base + linear dependency search for rank reduction."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
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

def _try_remove_terms(U, V, W, n, m, p, target_rank, time_limit=10.0):
    """
    Try to reduce rank from current to target_rank by finding linear dependencies.
    Uses iterative removal: try removing each term and check if remaining still reconstructs.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Work with copies
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    # Try to remove terms one by one
    removed = []
    for attempt in range(current_rank - target_rank):
        if time.time() - start_time > time_limit:
            break
        found_removal = False
        current_rank = U_curr.shape[1]
        
        for r in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            # Try removing term r
            U_test = np.delete(U_curr, r, axis=1)
            V_test = np.delete(V_curr, r, axis=1)
            W_test = np.delete(W_curr, r, axis=1)
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-6):
                U_curr = U_test
                V_curr = V_test
                W_curr = W_test
                found_removal = True
                break
        
        if not found_removal:
            break
    
    return U_curr, V_curr, W_curr

def _try_als_refine(U, V, W, n, m, p, target_rank, time_limit=20.0):
    """
    Use ALS to refine the decomposition to target_rank.
    Initialize from the structured decomposition by keeping the most important terms.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    T = _build_target_tensor(n, m, p)
    
    # Try multiple random initializations based on the structured decomposition
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    for trial in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly select target_rank columns from the structured decomposition
        if current_rank > target_rank:
            indices = np.random.choice(current_rank, target_rank, replace=False)
            U_init = U[:, indices].copy()
            V_init = V[:, indices].copy()
            W_init = W[:, indices].copy()
        else:
            U_init = U.copy()
            V_init = V.copy()
            W_init = W.copy()
        
        # Add small random perturbation to escape local minima
        U_init += np.random.randn(*U_init.shape) * 0.01
        V_init += np.random.randn(*V_init.shape) * 0.01
        W_init += np.random.randn(*W_init.shape) * 0.01
        
        # ALS iterations
        U_als = U_init.copy()
        V_als = V_init.copy()
        W_als = W_init.copy()
        
        for it in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for r in range(target_rank):
                num = np.zeros(n*m)
                den = np.zeros(n*m)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            num[i*m+j] += T[i,j,k] * V_als[j*p+k, r] * W_als[k*n+i, r]
                            den[i*m+j] += V_als[j*p+k, r]**2 * W_als[k*n+i, r]**2
                mask = den > 1e-12
                U_als[mask, r] = num[mask] / den[mask]
            
            # Update V
            for r in range(target_rank):
                num = np.zeros(m*p)
                den = np.zeros(m*p)
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            num[j*p+k] += T[i,j,k] * U_als[i*m+j, r] * W_als[k*n+i, r]
                            den[j*p+k] += U_als[i*m+j, r]**2 * W_als[k*n+i, r]**2
                mask = den > 1e-12
                V_als[mask, r] = num[mask] / den[mask]
            
            # Update W
            for r in range(target_rank):
                num = np.zeros(p*n)
                den = np.zeros(p*n)
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            num[k*n+i] += T[i,j,k] * U_als[i*m+j, r] * V_als[j*p+k, r]
                            den[k*n+i] += U_als[i*m+j, r]**2 * V_als[j*p+k, r]**2
                mask = den > 1e-12
                W_als[mask, r] = num[mask] / den[mask]
            
            # Check convergence
            T_recon = np.zeros((n, m, p))
            for r in range(target_rank):
                u = U_als[:, r].reshape(n, m)
                v = V_als[:, r].reshape(m, p)
                w = W_als[:, r].reshape(p, n)
                T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
            
            if np.max(np.abs(T_recon - T)) < 1e-8:
                break
        
        # Round to half-integers and verify
        U_rounded = np.round(U_als * 2) / 2.0
        V_rounded = np.round(V_als * 2) / 2.0
        W_rounded = np.round(W_als * 2) / 2.0
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-4):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            best_rank = target_rank
            break
        elif _verify_decomposition(U_als, V_als, W_als, n, m, p, tol=1e-6):
            # Accept even if not half-integer, but prefer half-integer
            if best_rank > target_rank:
                best_U, best_V, best_W = U_als, V_als, W_als
                best_rank = target_rank
    
    return best_U, best_V, best_W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44, V44, W44
    best_rank = 44
    
    # Verify the base decomposition
    if not _verify_decomposition(U44, V44, W44, n, m, p, tol=1e-6):
        # Fallback to schoolbook
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
    
    # Try to reduce rank using term removal (fast, deterministic)
    if time.time() < deadline - 30:
        U_rem, V_rem, W_rem = _try_remove_terms(U44, V44, W44, n, m, p, 40, time_limit=10.0)
        if U_rem.shape[1] < best_rank:
            best_U, best_V, best_W = U_rem, V_rem, W_rem
            best_rank = U_rem.shape[1]
    
    # Try ALS refinement for lower ranks
    for target_rank in range(best_rank - 1, 35, -1):
        if time.time() > deadline - 20:
            break
        
        U_als, V_als, W_als = _try_als_refine(
            best_U, best_V, best_W, n, m, p, target_rank, time_limit=15.0
        )
        
        if U_als.shape[1] < best_rank:
            best_U, best_V, best_W = U_als, V_als, W_als
            best_rank = U_als.shape[1]
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
