"""Rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import numpy as np
import itertools
import time

DIMENSIONS = [3, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_left():
    return START_TIME + TIME_LIMIT - SAFETY_MARGIN - time.time()

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] = 1 (for the 3x3*3x5 case).
    Convention: U indexed by (i,j) [n*m], V by (j,k) [m*p], W by (k,i) [p*n].
    T[i,j,k] = delta[i,i] * delta[j,j] * delta[k,k] = 1 for all valid indices."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == T[i,j,k]."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U shape (n*m, r), V (m*p, r), W (p*n, r)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # Reconstruct T
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, T, atol=tol, rtol=tol)

def _snap_to_halfint(M, tol=1e-6):
    """Snap matrix entries to nearest half-integer if close."""
    M2 = 2.0 * M
    Msnapped = np.round(M2) / 2.0
    if np.allclose(M2, Msnapped * 2, atol=tol, rtol=tol):
        return Msnapped
    return M

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Start with schoolbook as fallback
    rank_school = n * m * p  # 45
    best = None
    
    # Try to find lower rank decompositions
    # Known: 3x3x3 has rank 23 (Strassen-like), but 3x3x5 is different.
    # Let's try various rank values from low to high.
    
    # We'll use a structured search: try to find decompositions of rank r
    # by solving the nonlinear system with random initializations.
    
    # For each candidate rank, try multiple random initializations
    # and use gradient-free local search (or least-squares refinement).
    
    for rank in range(20, rank_school + 1):
        if _time_left() < 10:
            break
        
        found = False
        num_tries = min(50, max(5, int(200 * (rank_school / rank))))
        
        for attempt in range(num_tries):
            if _time_left() < 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Scale to reasonable magnitudes
            U = U / np.sqrt(rank)
            V = V / np.sqrt(rank)
            W = W / np.sqrt(rank)
            
            # Alternating least squares: fix two, solve for the third
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            converged = False
            for it in range(200):
                if _time_left() < 3:
                    break
                
                # Solve for U given V, W
                # For each (i,j): T[i,j,:] = sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c])
                # Let's define for fixed (i,j): target vector over k
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                
                # Build linear system for U[i*m+j,:]
                # For each k: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = T[i,j,k]
                # This is a linear system in U[i*m+j,:] of size rank x rank
                
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # A[c,k] = V[j*p+k,c] * W[k*n+i,c]
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V3[j, k, :] * W3[k, i, :]
                        # Solve A @ x = T[i,j,:] where x = U[i*m+j,:]
                        try:
                            sol, residuals, rankA, sv = np.linalg.lstsq(A, T[i, j, :], rcond=None)
                            new_U[i * m + j, :] = sol
                        except:
                            new_U[i * m + j, :] = U[i * m + j, :]
                
                U = new_U
                U3 = U.reshape(n, m, rank)
                
                # Solve for V given U, W
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # For fixed (j,k): sum_c V[j*p+k,c] * (U[:,j,c] * W[k,:,c]) = T[:,j,k]
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U3[i, j, :] * W3[k, i, :]
                        try:
                            sol, residuals, rankA, sv = np.linalg.lstsq(A, T[:, j, k], rcond=None)
                            new_V[j * p + k, :] = sol
                        except:
                            new_V[j * p + k, :] = V[j * p + k, :]
                
                V = new_V
                V3 = V.reshape(m, p, rank)
                
                # Solve for W given U, V
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # For fixed (k,i): sum_c W[k*n+i,c] * (U[i,:,c] * V[:,k,c]) = T[i,:,k]
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U3[i, j, :] * V3[j, k, :]
                        try:
                            sol, residuals, rankA, sv = np.linalg.lstsq(A, T[i, :, k], rcond=None)
                            new_W[k * n + i, :] = sol
                        except:
                            new_W[k * n + i, :] = W[k * n + i, :]
                
                W = new_W
                W3 = W.reshape(p, n, rank)
                
                # Check convergence
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.linalg.norm(recon - T)
                if err < 1e-8:
                    converged = True
                    break
            
            if converged or _time_left() < 3:
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                if np.allclose(recon, T, atol=1e-6, rtol=1e-6):
                    # Try to snap to half-integers
                    U_snapped = _snap_to_halfint(U)
                    V_snapped = _snap_to_halfint(V)
                    W_snapped = _snap_to_halfint(W)
                    
                    # Verify snapped version
                    U3s = U_snapped.reshape(n, m, rank)
                    V3s = V_snapped.reshape(m, p, rank)
                    W3s = W_snapped.reshape(p, n, rank)
                    recon_s = np.einsum('ijc,jkc,kic->ijk', U3s, V3s, W3s)
                    
                    if np.allclose(recon_s, T, atol=1e-6, rtol=1e-6):
                        U, V, W = U_snapped, V_snapped, W_snapped
                    else:
                        # Use the non-snapped version if it's valid
                        pass
                    
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    best = {'U': U_list, 'V': V_list, 'W': W_list, 'rank': rank}
                    found = True
                    break
        
        if found:
            break
    
    if best is not None:
        return {'U': best['U'], 'V': best['V'], 'W': best['W']}
    
    # Fallback: schoolbook
    rank = rank_school
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
