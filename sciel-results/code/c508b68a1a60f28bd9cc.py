"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using integer/rational arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition."""
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
    return U, V, W

def _find_decomposition_via_svd(n, m, p, deadline):
    """
    Use SVD-based tensor decomposition to find a low-rank approximation,
    then round to integers and verify.
    """
    # Build the tensor T[i,j,k] = 1 if this is a valid multiplication term
    # Actually, the matrix multiplication tensor is T[i,j,k] = δ_{j,k} in the
    # sense that C[i,k] = sum_j A[i,j] * B[j,k]
    # The structure tensor has T[i,j,k] = 1 for all (i,j,k) that participate,
    # but the "value" is 1 for each (i,j,k) position.
    
    # The tensor we need to decompose is: for each (i,j,k), the coefficient is 1
    # if we're computing C[i,k] from A[i,j]*B[j,k]. So T[i,j,k] = 1 for all
    # i,j,k in range.
    
    T = np.ones((n, m, p))
    
    # Try different ranks
    for rank in range(15, 40):
        if time.time() > deadline:
            break
        
        # Use CP decomposition via alternating least squares
        # Initialize randomly
        U_np = np.random.randn(n * m, rank)
        V_np = np.random.randn(m * p, rank)
        W_np = np.random.randn(p * n, rank)
        
        # Normalize
        U_np /= np.linalg.norm(U_np)
        V_np /= np.linalg.norm(V_np)
        W_np /= np.linalg.norm(W_np)
        
        # Alternating minimization
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), minimize ||T[i,j,:] - V[j,:] ⊙ W[:,i]||^2
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed (i,j), this is U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:])
                    # We want to solve: min_U sum_k (T[i,j,k] - U dot (V_k * W_k))^2
                    # This is a linear system: (V*W)^T (V*W) U = (V*W)^T T[i,j,:]
                    
                    # Build the matrix
                    M = np.zeros((rank, rank))
                    b = np.zeros(rank)
                    for k in range(p):
                        v_w = V_np[j * p + k] * W_np[k * n + i]
                        M += np.outer(v_w, v_w)
                        b += T[i, j, k] * v_w
                    
                    try:
                        U_np[i * m + j] = np.linalg.solve(M + 1e-8 * np.eye(rank), b)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, rank))
                    b = np.zeros(rank)
                    for i in range(n):
                        u_w = U_np[i * m + j] * W_np[k * n + i]
                        M += np.outer(u_w, u_w)
                        b += T[i, j, k] * u_w
                    
                    try:
                        V_np[j * p + k] = np.linalg.solve(M + 1e-8 * np.eye(rank), b)
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, rank))
                    b = np.zeros(rank)
                    for j in range(m):
                        u_v = U_np[i * m + j] * V_np[j * p + k]
                        M += np.outer(u_v, u_v)
                        b += T[i, j, k] * u_v
                    
                    try:
                        W_np[k * n + i] = np.linalg.solve(M + 1e-8 * np.eye(rank), b)
                    except:
                        pass
        
        # Round to nearest integers and verify
        U_int = np.round(U_np).astype(int)
        V_int = np.round(V_np).astype(int)
        W_int = np.round(W_np).astype(int)
        
        U_list = U_int.tolist()
        V_list = V_int.tolist()
        W_list = W_int.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list, rank
        
        # Also try rounding to half-integers
        U_half = np.round(U_np * 2) / 2
        V_half = np.round(V_np * 2) / 2
        W_half = np.round(W_np * 2) / 2
        
        U_list2 = U_half.tolist()
        V_list2 = V_half.tolist()
        W_list2 = W_half.tolist()
        
        if _verify_decomposition(U_list2, V_list2, W_list2, n, m, p):
            return U_list2, V_list2, W_list2, rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try SVD-based approach
    result = _find_decomposition_via_svd(n, m, p, deadline)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    # Final verification
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
