"""
Rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 6 matrix.

Strategy:
1. Try a few small ranks (starting from a lower bound) with random initializations.
2. Use a tensor network decomposition approach: we want to find matrices U, V, W
   such that for all i,j,k: 
       T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
   where T[i,j,k] = 1 if we're doing standard matrix multiplication.
   
   Wait, let me reconsider the tensor structure.
   
   For matrix multiplication C = A * B where A is n x m and B is m x p:
   C[i,k] = sum_j A[i,j] * B[j,k]
   
   The multiplication tensor has entries T[i,j,k] = 1 for all i,j,k (it's the identity 
   in the middle dimension). We want to decompose this as:
   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
   
   Wait, the task says:
   - U has shape (n*m, rank)
   - V has shape (m*p, rank)  
   - W has shape (p*n, rank)
   - W follows convention W[k*n+i, r]
   
   So the decomposition is:
   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
   
   And T[i,j,k] = 1 for all i,j,k (since matrix multiplication tensor is all ones).
   
   Actually wait - the matrix multiplication tensor for multiplying n x m by m x p:
   The tensor T is defined such that T[i,j,k] = 1 for all valid i,j,k.
   Because C[i,k] = sum_j A[i,j] * B[j,k], and the "structure" of the multiplication
   is captured by having T[i,j,k] = 1.
   
   So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

   This is a bilinear (actually trilinear) decomposition problem.

   Known lower bounds:
   - For n x m by m x p, the rank is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
   - Also at least n*m*p / max(n,m,p) ... no that's not right.
   - The trivial upper bound is n*m*p = 90.
   
   Let me think about what ranks are achievable.
   
   For square n x n by n x n, the best known is around n^2.5 for large n (Strassen-like).
   For 3x5 by 5x6, let's think...
   
   Actually, let me just try to find a good decomposition via optimization.
   
   Approach: Use gradient-based optimization or iterative refinement.
   
   Let me try a different approach: use the fact that we can fix one matrix and 
   solve for the others, then iterate.
   
   Or: use random initialization and gradient descent on the loss:
   L = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
   
   with constraints that entries are integers or half-integers.
   
   Actually, the problem says "integer/half-integer entries". So entries can be 
   ..., -1, -0.5, 0, 0.5, 1, ...
   
   This is a discrete optimization problem, which is hard. But maybe we can find 
   a solution with small integer entries.
   
   Let me try a simpler approach: use known algorithms or try small ranks with 
   random search.
   
   For rank R, we have R variables in each of U, V, W.
   Total unknowns: R*(n*m + m*p + p*n) = R*(15 + 30 + 18) = R*63
   
   Constraints: n*m*p = 90 equations.
   
   So for R >= 90/63 ≈ 1.43, we have more unknowns than equations. But the 
   equations are nonlinear (trilinear), so this doesn't directly help.
   
   Let me try to use a randomized approach: initialize U, V, W randomly with 
   small values, then use alternating least squares or similar.
   
   Actually, for the trilinear form, we can do:
   - Fix U and V, then for each (i,j,k), we have:
     sum_r W[k*n+i, r] * (U[i*m+j,r] * V[j*p+k,r]) = 1
   
   This is linear in W! So we can solve for W given U and V.
   
   Similarly, fix U and W, solve for V; fix V and W, solve for U.
   
   This is alternating least squares (ALS) for the trilinear decomposition.
   
   Let me implement this:
   
   1. Initialize U, V, W randomly.
   2. Iterate:
      a. Fix U, V. For each row w of W (corresponding to (k,i)), solve:
         sum_r W[k*n+i, r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
         This is a system of m equations in R unknowns for each (k,i).
         We can use least squares.
      b. Similarly for V and U.
   3. After convergence, round to nearest half-integer and verify.
   
   But this might not converge to a good solution. Let me also try random restarts.
   
   Given the time limit of 150s, I'll implement this with a deadline guard.
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 6]

def _make_tensor_indices(n, m, p):
    """Generate all (i,j,k) indices."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly satisfies T[i,j,k] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :]) @ W[k*n+i, :]
                # Actually, the decomposition is sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(W.shape[1]))
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def _als_step_U(U, V, W, n, m, p, rank):
    """Update U given fixed V and W."""
    U_new = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            # For fixed (i,j), we need:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # Let A[k,r] = V[j*p+k,r] * W[k*n+i,r]
            # Then sum_r U[i*m+j,r] * A[k,r] = 1 for all k
            # This is a linear system: A @ U[i*m+j,:] = ones(p)
            A = np.zeros((p, rank))
            for k in range(p):
                for r in range(rank):
                    A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Solve A @ x = ones(p) in least squares sense
            try:
                x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new[i*m+j, :] = x
            except:
                U_new[i*m+j, :] = U[i*m+j, :]
    return U_new

def _als_step_V(U, V, W, n, m, p, rank):
    """Update V given fixed U and W."""
    V_new = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            # For fixed (j,k), we need:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            # Let A[i,r] = U[i*m+j,r] * W[k*n+i,r]
            # Then sum_r V[j*p+k,r] * A[i,r] = 1 for all i
            A = np.zeros((n, rank))
            for i in range(n):
                for r in range(rank):
                    A[i, r] = U[i*m+j, r] * W[k*n+i, r]
            try:
                x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_new[j*p+k, :] = x
            except:
                V_new[j*p+k, :] = V[j*p+k, :]
    return V_new

def _als_step_W(U, V, W, n, m, p, rank):
    """Update W given fixed U and V."""
    W_new = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            # For fixed (k,i), we need:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            # Let A[j,r] = U[i*m+j,r] * V[j*p+k,r]
            # Then sum_r W[k*n+i,r] * A[j,r] = 1 for all j
            A = np.zeros((m, rank))
            for j in range(m):
                for r in range(rank):
                    A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            try:
                x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W_new[k*n+i, :] = x
            except:
                W_new[k*n+i, :] = W[k*n+i, :]
    return W_new

def _round_to_half_integer(matrix):
    """Round each entry to the nearest half-integer."""
    return np.round(np.array(matrix) * 2) / 2

def _compute_loss(U, V, W, n, m, p):
    """Compute the loss: sum over all (i,j,k) of (sum_r U*V*W - 1)^2."""
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(W.shape[1]))
                loss += (val - 1.0) ** 2
    return loss

def _try_rank(rank, n, m, p, max_iters=50, n_restarts=5, deadline=None):
    """Try to find a decomposition of the given rank using ALS."""
    best_solution = None
    best_loss = float('inf')
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Random initialization
        scale = 1.0 / (rank ** 0.5)
        U = np.random.randn(n*m, rank) * scale
        V = np.random.randn(m*p, rank) * scale
        W = np.random.randn(p*n, rank) * scale
        
        # ALS iterations
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            U = _als_step_U(U, V, W, n, m, p, rank)
            V = _als_step_V(U, V, W, n, m, p, rank)
            W = _als_step_W(U, V, W, n, m, p, rank)
            
            # Compute loss
            loss = _compute_loss(U, V, W, n, m, p)
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
            
            # Check convergence
            if loss < 1e-12:
                break
        
        # Try rounding to half-integers and verify
        if best_solution:
            U_r, V_r, W_r = best_solution
            U_half = _round_to_half_integer(U_r)
            V_half = _round_to_half_integer(V_r)
            W_half = _round_to_half_integer(W_r)
            
            valid, err = _verify_decomposition(U_half, V_half, W_half, n, m, p, tol=1e-9)
            if valid:
                return U_half, V_half, W_half
    
    # If no exact solution found, return the best we have (even if not exactly valid)
    if best_solution:
        U_r, V_r, W_r = best_solution
        U_half = _round_to_half_integer(U_r)
        V_half = _round_to_half_integer(V_r)
        W_half = _round_to_half_integer(W_r)
        return U_half, V_half, W_half
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to leave 10s margin
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Trivial upper bound: n*m*p = 90
    # Try ranks from 30 to 90
    
    # First, try to find a solution with rank 30 (the lower bound)
    for rank in range(30, 91):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, n, m, p, max_iters=30, n_restarts=3, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Verify
            valid, err = _verify_decomposition(U, V, W, n, m, p, tol=1e-9)
            if valid:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we couldn't find a valid solution, fall back to schoolbook
    rank = n * m * p
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
