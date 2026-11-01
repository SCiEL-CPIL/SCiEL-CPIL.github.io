"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    The multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A
    multiplied by the (j,k) entry of B contributes to the (i,k) entry of C.
    
    We seek rank-r factors U (nm x r), V (mp x r), W (pn x r) such that:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For n=2, m=2, p=6: nm=4, mp=12, pn=12, tensor size 4*12*12 = 576 entries.
    The schoolbook rank is 4*12 = 48. We search for lower ranks.
    
    Strategy: Try ranks from a lower bound upward. For each rank, use random
    initialization with alternating least squares to try to find an exact solution.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the multiplication tensor T
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # In our indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] -> C[i,k]
                # This is always a valid contribution
                T[i, j, k] = 1.0
    
    # Reshape T to (nm, mp, pn) for the tensor decomposition
    # T_reshaped[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # But wait - the tensor entries are only 1 for valid (i,j,k) triples, which is all of them.
    # Actually, the tensor for matrix multiplication is:
    # T[a, b, c] where a = i*m+j (row of A, col of A = row of B), 
    # b = j*p+k (row of B, col of B), c = k*n+i (col of C, row of C)
    # T[a,b,c] = 1 if the indices are consistent (j matches between a and b, i matches between a and c, k matches between b and c)
    
    # Let me rebuild properly
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j  # index in U
                b = j * p + k  # index in V
                c = k * n + i  # index in W
                T_full[a, b, c] = 1.0
    
    # The rank of this tensor is what we're looking for
    # Lower bound: max(nm, mp, pn) = max(4, 12, 12) = 12
    # Upper bound (schoolbook): nm * p = 4 * 6 = 24... wait, schoolbook is n*m*p = 2*2*6 = 24? 
    # No, schoolbook for n x m times m x p is n*m*p = 2*2*6 = 24
    # But the seed program uses rank = n*m*p = 24
    # Actually wait, let me re-read the seed: rank = n*m*p = 2*2*6 = 24
    # But nm = n*m = 4, mp = m*p = 12, pn = p*n = 12
    # Schoolbook: for each (i,j,k) triple, one term. That's n*m*p = 24 terms.
    
    # So schoolbook rank is 24. We want to find if we can do better.
    
    # Known results: For 2x2 x 2x6, the rank might be reducible.
    # Let's try to find a lower rank decomposition.
    
    # Try ranks from lower bound to schoolbook
    start_rank = max(nm, mp, pn)  # = 12
    end_rank = n * m * p  # = 24 (schoolbook)
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # First, let's try to verify if rank 24 (schoolbook) works and return it as fallback
    # Then try lower ranks
    
    # Try to find a solution with rank < 24
    # Use random initialization + alternating least squares
    
    def try_rank(rank, n_trials=50, seed=None):
        """Try to find a rank-r decomposition via random init + ALS."""
        if seed is not None:
            np.random.seed(seed)
        
        T_flat = T_full.reshape(nm, mp * pn)
        
        for trial in range(n_trials):
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale to avoid numerical issues
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)
            
            # Alternating least squares
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, T_full[a, :, :] = sum_r U[a,r] * (V[:,r] outer W[:,r])
                # T_full[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: T_full[a, :, :] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # Vectorize: T_flat[a, b*p+n+c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let M_U[b*p+n+c, r] = V[b,r] * W[c,r]
                # Then T_flat[a, :] = U[a, :] @ M_U.T
                # So U[a, :] = T_flat[a, :] @ M_U @ (M_U.T @ M_U)^{-1}... 
                # Actually: T_flat[a,:] = U[a,:] @ M_U.T => U[a,:] = T_flat[a,:] @ M_U @ inv(M_U.T @ M_U)
                # Or solve: M_U.T @ U[a,:].T = T_flat[a,:].T
                
                M_U = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        M_U[b * pn + c, :] = V[b, :] * W[c, :]
                
                # Solve for U
                try:
                    A_U = M_U.T @ M_U  # (rank, rank)
                    B_U = M_U.T @ T_flat.T  # (rank, nm)
                    U = np.linalg.solve(A_U, B_U).T
                except np.linalg.LinAlgError:
                    U = U + 0.01 * np.random.randn(nm, rank)
                
                # Update V: for each b, T_full[:, b, :] = sum_r V[b,r] * (U[:,r] outer W[:,r])
                M_V = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        M_V[a * pn + c, :] = U[a, :] * W[c, :]
                
                T_flat_V = T_full.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    A_V = M_V.T @ M_V
                    B_V = M_V.T @ T_flat_V.T
                    V = np.linalg.solve(A_V, B_V).T
                except np.linalg.LinAlgError:
                    V = V + 0.01 * np.random.randn(mp, rank)
                
                # Update W: for each c, T_full[:, :, c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
                M_W = np.zeros((nm * mp, rank))
                for a in range(nm):
                    for b in range(mp):
                        M_W[a * mp + b, :] = U[a, :] * V[b, :]
                
                T_flat_W = T_full.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    A_W = M_W.T @ M_W
                    B_W = M_W.T @ T_flat_W.T
                    W = np.linalg.solve(A_W, B_W).T
                except np.linalg.LinAlgError:
                    W = W + 0.01 * np.random.randn(pn, rank)
                
                # Check convergence
                error = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (val - T_full[a, b, c]) ** 2
                
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Verify the solution
                max_err = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            max_err = max(max_err, abs(val - T_full[a, b, c]))
                
                if max_err < 1e-9:
                    # Convert to lists
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    # Try ranks from lower to higher
    for rank in range(start_rank, end_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, n_trials=30, seed=rank * 1000)
        if result is not None:
            best_result = result
            break
    
    # If no lower rank found, return schoolbook
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
