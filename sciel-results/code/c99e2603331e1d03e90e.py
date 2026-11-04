"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    The multiplication tensor T has T[i,j,k] = 1 if the (i,j) element of the first
    matrix and (j,k) element of the second matrix contribute to the (i,k) element
    of the product. We seek U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    such that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    Strategy: Start with schoolbook rank=80, then try to reduce rank by finding
    dependencies or using algebraic structure. Use a deadline guard to ensure
    completion within time limits.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin before 150s limit
    
    # Build the multiplication tensor T as a 3D array
    # T[i, j, k] = 1 if (i,j) in first matrix and (j,k) in second contribute to (i,k)
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # Every triple (i,j,k) where j is shared contributes
    
    # The tensor can be viewed as a matrix of shape (n*m, m*p) with entries
    # M[i*m+j, j*p+k] = T[i,j,k]
    # But more precisely, for rank decomposition we need:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3-way tensor decomposition.
    
    # Schoolbook: rank = n*m*p = 80
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if verify(U_school, V_school, W_school):
        best_rank = rank_schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find a lower rank decomposition
    # The key insight: the multiplication tensor for m x n * n x p has a known
    # structure. For our case (4x4)*(4x5), we can try various approaches.
    
    # Approach 1: Try to find linear dependencies among the schoolbook terms
    # to reduce rank. This is equivalent to finding a rank decomposition with
    # fewer terms.
    
    # For matrix multiplication, the tensor T_{ijk} = delta_{j,j} (trivially 1 for
    # all valid indices in this formulation). The rank of the multiplication tensor
    # is related to the bilinear complexity of matrix multiplication.
    
    # Known results: For 4x4 * 4x5, the minimal rank is not known to be less than
    # the schoolbook 80 in general, but let's try to find improvements.
    
    # Approach 2: Use the fact that we can permute and combine terms.
    # Try a randomized search for lower rank decompositions.
    
    # First, let's try a structured approach based on the algebraic structure.
    # The tensor T can be seen as having a "slice" structure. For each j (the
    # shared index), we have a rank-1 contribution from the j-th row of the
    # first matrix and j-th column of the second.
    
    # Let's try to find if there's a decomposition with rank < 80 by solving
    # a system of equations.
    
    # For a given rank r, we need to find U, V, W such that:
    # sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of n*m*p = 80 equations with r*(n*m + m*p + p*n) = r*(16+20+20) = 56r unknowns.
    # For r < 80, we have fewer unknowns than equations, so we need to be clever.
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms.
    
    # Actually, let's try a different approach: use the known structure of matrix
    # multiplication tensors. The tensor for multiplying an m x n matrix by an
    # n x p matrix has a natural decomposition into n rank-(m*p) tensors, one for
    # each value of the shared index j.
    
    # For our case, n=4 (shared dimension), so we have 4 slices, each of rank 4*5=20.
    # Total schoolbook rank = 4*20 = 80.
    
    # Can we do better? The answer depends on the bilinear complexity of 4x4 * 4x5
    # matrix multiplication. This is a specific instance and the optimal rank is
    # not trivially known.
    
    # Let's try a numerical approach: use SVD or similar to find a low-rank
    # approximation, then check if it's exact.
    
    # Reshape T into a matrix for SVD. The tensor T[i,j,k] can be matricized as
    # T_{(i,j), k} or T_{i, (j,k)} etc.
    
    # Matricization 1: T_{i, (j,k)} - shape (n, m*p)
    T_mat1 = T.reshape(n, m*p)
    # Matricization 2: T_{(i,j), k} - shape (n*m, p)
    T_mat2 = T.reshape(n*m, p)
    # Matricization 3: T_{(j,k), i} - shape (m*p, n)
    T_mat3 = T.reshape(m*p, n)
    
    # The rank of the tensor is at least the maximum of the ranks of its matricizations.
    # Let's compute these ranks.
    rank1 = np.linalg.matrix_rank(T_mat1)
    rank2 = np.linalg.matrix_rank(T_mat2)
    rank3 = np.linalg.matrix_rank(T_mat3)
    
    # For our tensor, all entries are 1, so:
    # T_mat1 is a 4x20 matrix of all 1s, rank = 1
    # T_mat2 is a 16x5 matrix of all 1s, rank = 1
    # T_mat3 is a 20x4 matrix of all 1s, rank = 1
    # So the tensor rank is at least 1, but this doesn't help much.
    
    # The issue is that the tensor decomposition we need is a 3-way decomposition
    # with the specific structure where the indices are coupled.
    
    # Let's try a direct numerical search for a low-rank decomposition.
    # We'll use a simple iterative method.
    
    def try_rank_decomposition(r, max_iter=1000, tol=1e-6):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < tol:
                return best_U, best_V, best_W
            
            # Update: gradient descent
            lr = 0.01
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual = 1.0 - np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        dU[i*m+j, :] += residual * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += residual * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += residual * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < tol:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from 60 down to 1
    found = False
    for r in range(60, 0, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_decomposition(r, max_iter=500)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
    
    # If no better rank found, return schoolbook
    if not found:
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
