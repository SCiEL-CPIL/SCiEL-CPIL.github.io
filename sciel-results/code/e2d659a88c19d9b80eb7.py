"""
Program for Matrix Multiplication 3x4x8 Tensor Decomposition.
Uses alternating minimization with exact verification to find low-rank decompositions.
"""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the U, V, W matrices for a low-rank exact decomposition.
    Uses alternating minimization and exact verification.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Best result so far: schoolbook decomposition
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Lower bound for rank
    lower_bound = max(n * m, m * p, p * n)
    
    # Try to find lower-rank decompositions
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try alternating minimization for this rank
        result = _alternating_minimization(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
            # Found a decomposition, continue searching for lower ranks
            # But we can't go below lower_bound, so we're done if at lower_bound
            if target_rank == lower_bound:
                break
    
    # Convert to lists for the return interface
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return (U, V, W)

def _alternating_minimization(n, m, p, target_rank, deadline):
    """
    Try to find an exact decomposition of the given rank using alternating minimization.
    Returns (U, V, W) as numpy arrays or None if not found.
    """
    np.random.seed(42 + target_rank)
    
    # Multiple random restarts
    max_restarts = 50
    max_iterations = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            break
        
        # Initialize with random matrices
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Alternating minimization
        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            # Fix U and V, solve for W
            W = _solve_for_W(n, m, p, target_rank, U, V)
            if W is None:
                break
            
            # Fix V and W, solve for U
            U = _solve_for_U(n, m, p, target_rank, V, W)
            if U is None:
                break
            
            # Fix U and W, solve for V
            V = _solve_for_V(n, m, p, target_rank, U, W)
            if V is None:
                break
            
            # Check convergence: verify the solution
            if _verify_solution(n, m, p, U, V, W):
                # Found a valid solution!
                # Round to nearest half-integer if possible
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify_solution(n, m, p, U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
                else:
                    # Return the unrounded solution
                    return (U, V, W)
    
    return None

def _solve_for_W(n, m, p, target_rank, U, V):
    """
    Solve for W given U and V.
    For each (i,k): sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    This is linear in W.
    """
    W = np.zeros((p * n, target_rank))
    
    # For each output position (i,k), we have a linear equation in W[k*n+i, :]
    # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # But this is not independent for each (i,k) - we need the full system
    # Actually, for fixed U and V, the equations for different (i,k) are independent
    # because W[k*n+i, r] only appears in the equation for that specific (i,k)
    
    # Wait, that's not right. Let me reconsider.
    # The equation is: for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # For fixed (i,k), summing over j: sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # No, that's not how it works. The equation must hold for EACH (i,j,k) separately.
    
    # So for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This means: for each (i,k), and for each j: sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    # So W[k*n+i, :] must satisfy: for all j, (U[i*m+j,:] * V[j*p+k,:]) · W[k*n+i,:] = 1
    
    # This is a system of m equations in target_rank unknowns for each (i,k)
    # If target_rank >= m, we can solve it
    
    for i in range(n):
        for k in range(p):
            # Build the system: for each j, compute the vector (U[i*m+j,:] * V[j*p+k,:])
            # and we need its dot product with W[k*n+i,:] to be 1
            
            # This is an underdetermined system if target_rank > m
            # We want the minimum-norm solution or a specific solution
            
            # Actually, we need: for all j, sum_r A[j,r] * W[k*n+i,r] = 1
            # where A[j,r] = U[i*m+j,r] * V[j*p+k,r]
            
            # This is: A @ W[k*n+i,:] = ones(m)
            # where A is m x target_rank
            
            A = np.zeros((m, target_rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            # Solve A @ w = ones(m) for w
            # If target_rank >= m, use least squares
            # If the system is consistent, we get an exact solution
            
            try:
                if target_rank >= m:
                    # Use least squares, then check if it's exact
                    w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if the solution is exact
                    if np.allclose(A @ w, np.ones(m), atol=1e-10):
                        W[k * n + i, :] = w
                    else:
                        # Not an exact solution, try a different approach
                        # Use the Moore-Penrose pseudoinverse
                        w = np.linalg.pinv(A) @ np.ones(m)
                        if np.allclose(A @ w, np.ones(m), atol=1e-10):
                            W[k * n + i, :] = w
                        else:
                            return None
                else:
                    # Underdetermined, use least squares
                    w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    if np.allclose(A @ w, np.ones(m), atol=1e-10):
                        W[k * n + i, :] = w
                    else:
                        return None
            except np.linalg.LinAlgError:
                return None
    
    return W

def _solve_for_U(n, m, p, target_rank, V, W):
    """
    Solve for U given V and W.
    For each (i,j): sum_k sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    For fixed (i,j): sum_k (V[j*p+k,:] * W[k*n+i,:]) · U[i*m+j,:] = 1
    """
    U = np.zeros((n * m, target_rank))
    
    for i in range(n):
        for j in range(m):
            # Build the system: for each k, compute the vector (V[j*p+k,:] * W[k*n+i,:])
            # and we need its dot product with U[i*m+j,:] to be 1
            
            A = np.zeros((p, target_rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            
            # Solve A @ u = ones(p) for u
            try:
                if target_rank >= p:
                    u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    if np.allclose(A @ u, np.ones(p), atol=1e-10):
                        U[i * m + j, :] = u
                    else:
                        u = np.linalg.pinv(A) @ np.ones(p)
                        if np.allclose(A @ u, np.ones(p), atol=1e-10):
                            U[i * m + j, :] = u
                        else:
                            return None
                else:
                    u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    if np.allclose(A @ u, np.ones(p), atol=1e-10):
                        U[i * m + j, :] = u
                    else:
                        return None
            except np.linalg.LinAlgError:
                return None
    
    return U

def _solve_for_V(n, m, p, target_rank, U, W):
    """
    Solve for V given U and W.
    For each (j,k): sum_i sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    For fixed (j,k): sum_i (U[i*m+j,:] * W[k*n+i,:]) · V[j*p+k,:] = 1
    """
    V = np.zeros((m * p, target_rank))
    
    for j in range(m):
        for k in range(p):
            # Build the system: for each i, compute the vector (U[i*m+j,:] * W[k*n+i,:])
            # and we need its dot product with V[j*p+k,:] to be 1
            
            A = np.zeros((n, target_rank))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            
            # Solve A @ v = ones(n) for v
            try:
                if target_rank >= n:
                    v, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    if np.allclose(A @ v, np.ones(n), atol=1e-10):
                        V[j * p + k, :] = v
                    else:
                        v = np.linalg.pinv(A) @ np.ones(n)
                        if np.allclose(A @ v, np.ones(n), atol=1e-10):
                            V[j * p + k, :] = v
                        else:
                            return None
                else:
                    v, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    if np.allclose(A @ v, np.ones(n), atol=1e-10):
                        V[j * p + k, :] = v
                    else:
                        return None
            except np.linalg.LinAlgError:
                return None
    
    return V

def _verify_solution(n, m, p, U, V, W):
    """
    Verify that the solution exactly decomposes the matrix multiplication tensor.
    For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    """
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    
    return True
# EVOLVE-BLOCK-END
