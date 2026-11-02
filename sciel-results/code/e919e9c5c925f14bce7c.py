"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    # Build the tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(T, 1.0, atol=tol)


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _try_fixed_U(n, m, p, target_rank, time_limit, seed=42):
    """
    Try to find a rank-`target_rank` decomposition by fixing U to have a specific
    sparse structure and solving for V and W.
    
    Strategy: For each rank r, we try different patterns for U.
    One effective pattern: U has 1s at positions that form a "grid" pattern.
    For 3x3x4, we can try U where each column has exactly 2 non-zero entries.
    
    Actually, let's use a more systematic approach:
    For each target rank R, we can try to find U, V, W such that the tensor equation holds.
    We fix U to have a specific structure and solve the resulting linear system for V and W.
    
    Key insight: If we fix U, the equation becomes:
    For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is still bilinear in V and W. But if we also fix V (or W), it becomes linear.
    
    Let's try: fix both U and V to sparse patterns, then solve for W linearly.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    num_entries = n * m * p
    
    # Try different target ranks
    # For each rank, try multiple random initializations of the sparse structure
    
    # Approach: 
    # 1. Generate random sparse U (n*m x R) with entries in {-1, 0, 1}
    # 2. Generate random sparse V (m*p x R) with entries in {-1, 0, 1}
    # 3. Solve for W (p*n x R) from the linear system:
    #    For each r, and for each (i,j,k):
    #    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    #    
    # This can be rewritten as: for each (k,i), 
    # sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    #
    # Let A[(k,i), r] = sum_j U[i*m+j,r] * V[j*p+k,r]
    # Then we need: A @ W.T = ones(n*p)  (where W is p*n x R, so W.T is R x p*n)
    # Actually: sum_r A[(k,i), r] * W[k*n+i, r] = 1 for all (k,i)
    # This is: A @ W^T = 1_vector, where A is (n*p) x R and W^T is R x (n*p)
    # So: W^T = A^{-1} @ 1_vector if A is square and invertible
    # But A is (n*p) x R, and we need R >= n*p for this to work in general.
    # For n=3, p=4, n*p=12. So R >= 12.
    
    # Let's try R from 12 up to 36
    best_result = None
    
    for R in range(12, 37):
        if time.time() > start + time_limit - 2:
            break
        
        # Try multiple random attempts for this rank
        for attempt in range(50):
            if time.time() > start + time_limit - 2:
                break
            
            # Generate sparse U: each column has 2-3 non-zero entries
            U = np.zeros((n * m, R))
            for r in range(R):
                # Choose 2 or 3 random positions
                num_nonzero = rng.choice([2, 3])
                positions = rng.choice(n * m, size=num_nonzero, replace=False)
                for pos in positions:
                    U[pos, r] = rng.choice([-1, 1])
            
            # Generate sparse V: each column has 2-3 non-zero entries
            V = np.zeros((m * p, R))
            for r in range(R):
                num_nonzero = rng.choice([2, 3])
                positions = rng.choice(m * p, size=num_nonzero, replace=False)
                for pos in positions:
                    V[pos, r] = rng.choice([-1, 1])
            
            # Compute A: A[(k,i), r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # A has shape (n*p, R)
            A = np.zeros((n * p, R))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        A[k * n + i, :] += U[i * m + j, :] * V[j * p + k, :]
            
            # Now solve: for each (k,i), sum_r A[k*n+i, r] * W[k*n+i, r] = 1
            # This is a system of n*p equations in n*p*R unknowns (W entries)
            # But it's separable by (k,i): for each (k,i), we have one equation with R unknowns
            # So we can't solve uniquely unless we add constraints.
            
            # Alternative: if R == n*p, we can try to make A invertible and set W^T = A^{-1}
            if R == n * p:
                try:
                    # Check if A is invertible
                    det = np.linalg.det(A)
                    if abs(det) > 1e-6:
                        # W^T = A^{-1} @ ones
                        ones = np.ones(n * p)
                        W_T = np.linalg.solve(A, ones)
                        W = W_T.T.reshape(p * n, R)
                        
                        # Verify
                        if _verify(U, V, W, n, m, p):
                            best_result = (U, V, W)
                            return best_result
                except np.linalg.LinAlgError:
                    pass
            else:
                # For R > n*p, the system is underdetermined.
                # We can use least squares or find a particular solution.
                # For each (k,i), we need: A[k*n+i, :] @ W[k*n+i, :] = 1
                # This is one linear equation in R unknowns for each (k,i).
                # We can find a particular solution using pseudo-inverse.
                try:
                    # For each row of A, find a vector w such that A[row] @ w = 1
                    W = np.zeros((p * n, R))
                    for ki in range(n * p):
                        a_row = A[ki, :]
                        # Find w such that a_row @ w = 1
                        # Minimum norm solution: w = a_row^T / (a_row @ a_row^T)
                        norm_sq = np.dot(a_row, a_row)
                        if norm_sq > 1e-9:
                            W[ki, :] = a_row / norm_sq
                        else:
                            # Can't satisfy this equation
                            break
                    else:
                        # All equations satisfied
                        if _verify(U, V, W, n, m, p):
                            best_result = (U, V, W)
                            return best_result
                except Exception:
                    pass
    
    return best_result


def _try_gradient_descent(n, m, p, target_rank, time_limit, seed=42):
    """
    Use gradient descent to find a rank-`target_rank` decomposition.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Initialize with random values
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    learning_rate = 0.01
    best_loss = float('inf')
    best_params = (U.copy(), V.copy(), W.copy())
    
    for step in range(10000):
        if time.time() > start + time_limit - 2:
            break
        
        # Compute tensor T
        T = np.zeros((n, m, p))
        for r in range(target_rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        # Compute loss
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_params = (U.copy(), V.copy(), W.copy())
        
        if loss < 1e-12:
            U, V, W = best_params
            if _verify(U, V, W, n, m, p):
                return (U, V, W)
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(target_rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = T[i, j, k] - 1.0
                        dU[i * m + j, r] += 2 * err * v[j, k] * w[k, i]
                        dV[j * p + k, r] += 2 * err * u[i, j] * w[k, i]
                        dW[k * n + i, r] += 2 * err * u[i, j] * v[j, k]
        
        # Update
        U -= learning_rate * dU
        V -= learning_rate * dV
        W -= learning_rate * dW
    
    U, V, W = best_params
    if _verify(U, V, W, n, m, p):
        return (U, V, W)
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p
    
    # Try to find a better decomposition using fixed-U approach
    # Start from lower ranks and work up
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 5:
            break
        
        remaining_time = deadline - time.time() - 2
        if remaining_time < 1:
            break
        
        result = _try_fixed_U(n, m, p, target_rank, remaining_time, seed=target_rank)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
    
    # If fixed-U didn't work, try gradient descent for a few ranks
    if best_rank == n * m * p:
        for target_rank in [15, 16, 18, 20]:
            if time.time() > deadline - 5:
                break
            
            remaining_time = deadline - time.time() - 2
            if remaining_time < 5:
                break
            
            result = _try_gradient_descent(n, m, p, target_rank, remaining_time, seed=target_rank)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = target_rank
                    break
    
    return best_result

# EVOLVE-BLOCK-END
