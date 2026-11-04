import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 5, 5]
DEADLINE_SECONDS = 145.0 # Safety margin

def _time_left():
    return DEADLINE_SECONDS - (time.time() - _START_TIME)

_START_TIME = time.time()

def _validate_np(U, V, W, n, m, p, tol=1e-9):
    """Fast validation using NumPy."""
    U3 = np.array(U, dtype=np.float64)
    V3 = np.array(V, dtype=np.float64)
    W3 = np.array(W, dtype=np.float64)
    
    # Construct the tensor contraction
    # For each i, k, we need sum_j A[i,j] B[j,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
    # This expands to sum_{j, j', r} W[k,i,r] U[i,j,r] V[j',k,r] A[i,j] B[j',k]
    # We require sum_{r} W[k,i,r] U[i,j,r] V[j',k,r] == delta_{j,j'}
    
    for i in range(n):
        for k in range(p):
            # Construct 25xR matrix M
            # M[j*5 + j', r] = U[i,j,r] * V[j',k,r]
            M = np.zeros((m*m, len(W3[0,0,:]) if W3.ndim==3 else W3.shape[1]), dtype=np.float64)
            # W3 shape is (p, n, R)
            R = W3.shape[2]
            M = np.zeros((m*m, R), dtype=np.float64)
            
            for j in range(m):
                for jp in range(m):
                    # Vector for this (j, jp) pair
                    # Sum over r of W[k,i,r] * U[i,j,r] * V[jp,k,r]
                    # We want this sum to be 1 if j==jp else 0
                    # So we are solving for W
                    # The check is if the W we found satisfies this.
                    pass
            
            # Actually, just compute the resulting tensor and compare to identity
            # Result[i, j, jp, k] = sum_r W[k,i,r] U[i,j,r] V[jp,k,r]
            # We want Result[i, j, jp, k] == 1 if j==jp else 0
            
            # Efficient check:
            # For fixed i, k:
            # Let U_i be (m, R), V_k be (m, R)
            # W_ik is (R,)
            # The condition is sum_r W_ik[r] U_i[j,r] V_k[jp,r] == delta_{j,jp}
            
            U_i = U3[i, :, :] # Shape (m, R) -> actually U is (n*m, R), so U3[i*m : (i+1)*m, :]
            # Wait, U shape is (n*m, R). Index i*m+j
            U_i = U3[i*m : (i+1)*m, :]
            V_k = V3[k*p : (k+1)*p, :] # Wait, V shape is (m*p, R). Index j*p+k
            # V is indexed by (j, k). So for fixed k, we need V[j, k] for all j.
            # V3[j*p + k, :]
            
            # Let's re-index properly
            # U: (n*m, R). Row index i*m + j.
            # V: (m*p, R). Row index j*p + k.
            # W: (p*n, R). Row index k*n + i.
            
            # For fixed i, k:
            # U_part: (m, R) where U_part[j, r] = U[i*m + j, r]
            # V_part: (m, R) where V_part[j, r] = V[j*p + k, r]
            # W_vec: (R,) where W_vec[r] = W[k*n + i, r]
            
            U_part = U3[i*m : (i+1)*m, :]
            # V_part needs to be transposed or gathered correctly
            # V indices: j*p + k
            # We want a matrix V_part of shape (m, R) where V_part[j, r] = V[j*p + k, r]
            V_part = np.array([V3[j*p + k, :] for j in range(m)])
            W_vec = W3[k*n + i, :]
            
            # Compute matrix C_ik of shape (m, m)
            # C_ik[j, jp] = sum_r W_vec[r] * U_part[j, r] * V_part[jp, r]
            # C_ik = U_part.T @ (W_vec * V_part.T) ? No.
            # C_ik = (U_part * W_vec[None, :]).T @ V_part
            # Or: np.einsum('jr, r, kr->jk', U_part, W_vec, V_part)
            
            C_ik = np.einsum('jr, r, pr->jp', U_part, W_vec, V_part)
            
            # Check against identity
            if not np.allclose(C_ik, np.eye(m), atol=tol):
                return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _search_decomposition(n, m, p, max_rank=75, min_rank=60, max_time=130):
    """
    Search for a decomposition with rank < max_rank.
    Uses random U, V and solves for W.
    """
    deadline = time.time() + max_time
    best_result = None
    
    # Try ranks from max_rank-1 down to min_rank
    for R in range(max_rank - 1, min_rank - 1, -1):
        if time.time() > deadline:
            break
            
        # Try a few random instances for this rank
        attempts = 50 if R > 70 else 100
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse U and V
            # Entries from {-1, 0, 1}
            # Density ~ 0.3
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            for idx in range(U.size):
                if random.random() < 0.3:
                    U.flat[idx] = random.choice([-1, 1])
            
            for idx in range(V.size):
                if random.random() < 0.3:
                    V.flat[idx] = random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p * n, R))
            valid = True
            for i in range(n):
                for k in range(p):
                    U_part = U[i*m : (i+1)*m, :]
                    V_part = np.array([V[j*p + k, :] for j in range(m)])
                    
                    # System: sum_r W[r] * U_part[j, r] * V_part[jp, r] = delta_{j,jp}
                    # Vectorize: 
                    # Let M be (m*m, R)
                    # M[j*m+jp, r] = U_part[j, r] * V_part[jp, r]
                    M = np.zeros((m*m, R))
                    for j in range(m):
                        for jp in range(m):
                            M[j*m+jp, :] = U_part[j, :] * V_part[jp, :]
                    
                    # Target vector b
                    b = np.zeros(m*m)
                    for j in range(m):
                        b[j*m+j] = 1.0
                    
                    # Solve M x = b
                    # Use least squares
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if x is "nice" (close to integers or half-integers)
                    # Snap to nearest 0.5
                    x_snapped = np.round(x * 2) / 2
                    
                    # Verify if x_snapped is an exact solution
                    # M @ x_snapped == b
                    if not np.allclose(M @ x_snapped, b, atol=1e-9):
                        # Try snapping to integers
                        x_int = np.round(x)
                        if not np.allclose(M @ x_int, b, atol=1e-9):
                            # Maybe larger denominators? Try 1/3?
                            x_third = np.round(x * 3) / 3
                            if not np.allclose(M @ x_third, b, atol=1e-9):
                                valid = False
                                break
                    
                    if valid:
                        # Use the snapped solution
                        W[k*n + i, :] = x_snapped
                        if not np.allclose(M @ x_snapped, b, atol=1e-9):
                             W[k*n + i, :] = x_int
                        if not np.allclose(M @ W[k*n + i, :], b, atol=1e-9):
                             W[k*n + i, :] = x_third
                        if not np.allclose(M @ W[k*n + i, :], b, atol=1e-9):
                             valid = False
                             break
            
            if valid:
                # Final full validation
                if _validate_np(U, V, W, n, m, p):
                    return U, V, W, R
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try to find better
    if _time_left() > 30:
        result = _search_decomposition(n, m, p, max_rank=rank, min_rank=60, max_time=_time_left() - 10)
        if result is not None:
            U, V, W, rank = result
    
    # Final verification
    if not _validate_np(U, V, W, n, m, p):
        U, V, W, rank = _schoolbook(n, m, p)
    
    # Convert to integer/half-integer lists
    # The problem states "integer/half-integer entries".
    # We snap to 0.5 precision in the search.
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2
    
    # Convert to lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
