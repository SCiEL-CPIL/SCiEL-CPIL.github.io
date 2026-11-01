"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition search."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    """Build the 2x3x8 matrix multiplication tensor T[i,j,k] = 1."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify(U, V, W, T, tol=1e-8):
    """Verify that U, V, W decompose T."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Compute reconstruction: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, T, atol=tol, rtol=tol)

def _als_solve(T, rank, max_iter=200, tol=1e-10, seed=42):
    """ALS search for rank-r decomposition of tensor T."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    # Initialize
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    T_flat = T.reshape(n * m * p)
    
    for it in range(max_iter):
        # Update U: for each (i,j), U[i*m+j, :] = V[j*p+k, :] * W[k*n+i, :] (summed over k)
        for i in range(n):
            for j in range(m):
                # U[i*m+j, c] = sum_k T[i,j,k] * V[j*p+k, c] * W[k*n+i, c] / (sum_k V[j*p+k,c]^2 * W[k*n+i,c]^2)
                # Actually, we want: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix V, W, solve for U:
                # For each c: U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) summed over k should give T[i,j,k]
                # This is: U[i*m+j,c] * sum_k (V[j*p+k,c] * W[k*n+i,c] * delta) ... 
                # Better: treat it as linear system in U[i*m+j,:]
                # T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                # For fixed (i,j), let A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                # Then T[i,j,:] = U[i*m+j,:] @ A (where A is p x rank)
                A = V[j*p:(j+1)*p, :] * W.reshape(p, n, rank)[:, i, :]  # (p, rank)
                # Solve: T[i,j,:] = U[i*m+j,:] @ A
                # U[i*m+j,:] = T[i,j,:] @ pinv(A)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, T[i, j, :], rcond=None)[0]
                except:
                    U[i*m+j, :] = T[i, j, :] @ np.linalg.pinv(A)
        
        # Update V: for each (j,k), V[j*p+k, c]
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix U, W, solve for V[j*p+k,:]
                # T[:,j,k] = U[:,j,:] * V[j*p+k,:] * W[k*:,i,:] (summed over i,c)
                # Let B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                # T[:,j,k] = V[j*p+k,:] @ B (B is n x rank)
                B = U.reshape(n, m, rank)[:, j, :] * W.reshape(p, n, rank)[k, :, :]  # (n, rank)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B.T, T[:, j, k], rcond=None)[0]
                except:
                    V[j*p+k, :] = T[:, j, k] @ np.linalg.pinv(B)
        
        # Update W: for each (k,i), W[k*n+i, c]
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix U, V, solve for W[k*n+i,:]
                # T[i,:,k] = U[i*m+,:,:] * V[:,p+k,:] * W[k*n+i,:] (summed over j,c)
                # Let C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # T[i,:,k] = W[k*n+i,:] @ C (C is m x rank)
                C = U.reshape(n, m, rank)[i, :, :] * V.reshape(m, p, rank)[:, k, :]  # (m, rank)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C.T, T[i, :, k], rcond=None)[0]
                except:
                    W[k*n+i, :] = T[i, :, k] @ np.linalg.pinv(C)
        
        # Check convergence
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
    
    return U, V, W, err

def _to_rational(U, V, W, denom=1000):
    """Convert float matrices to Fraction for exactness check."""
    def mat_to_frac(M):
        return [[Fraction(M[i][j]).limit_denominator(denom) for j in range(M.shape[1])] 
                for i in range(M.shape[0])]
    return mat_to_frac(U), mat_to_frac(V), mat_to_frac(W)

def _try_rank(rank, max_seeds=5, max_iter=300):
    """Try to find a rank-r decomposition."""
    T = _build_target_tensor()
    best_err = np.inf
    best_result = None
    for seed in range(max_seeds):
        U, V, W, err = _als_solve(T, rank, max_iter=max_iter, tol=1e-12, seed=seed)
        if err < best_err:
            best_err = err
            best_result = (U, V, W)
        if err < 1e-8:
            # Verify with exact fractions
            Uf, Vf, Wf = _to_rational(U, V, W, denom=10000)
            # Check exact decomposition
            n, m, p = DIMENSIONS
            T_exact = np.ones((n, m, p))
            recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(rank):
                            s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                        recon[i,j,k] = float(s)
            if np.allclose(recon, T_exact, atol=1e-10):
                return Uf, Vf, Wf, True
    return best_result, False

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # Known good: rank-44 construction (Strassen-style)
    # Fallback: schoolbook rank 48
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    
    # Build schoolbook as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank
    # Based on knowledge: rank 44 is achievable (Strassen 7 + 4 per 2x3x2 block)
    # Let's try ranks from 40 upward
    T = _build_target_tensor()
    
    for rank in [40, 41, 42, 43, 44, 45, 46, 47]:
        if time.time() > deadline - 10:
            break
        try:
            result, success = _try_rank(rank, max_seeds=3, max_iter=200)
            if success:
                # Found valid rank-r decomposition
                U, V, W = result
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass
    
    # Fallback to schoolbook
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
