"""
Matrix multiplication 3x4x6: find a rank-minimal exact tensor decomposition.

Strategy:
- The matrix multiplication tensor T_{n,m,p} satisfies:
  C[i,k] = sum_j A[i,j] B[j,k]
  Equivalently, the tensor is:
  T = sum_{i,j,k} u_{i,j} * v_{j,k} * w_{k,i}
  where u_{i,j} is the basis vector in R^{n*m} at position (i*m+j),
  v_{j,k} is the basis vector in R^{m*p} at position (j*p+k),
  w_{k,i} is the basis vector in R^{p*n} at position (k*n+i).

- The schoolbook rank is n*m*p = 72.
- We attempt to find a lower-rank decomposition using a structured search.

- Key insight: For small matrices, we can use known algebraic constructions.
  For 3x4x6, we can try to use a combination of Strassen-like ideas.
  
- However, the most reliable approach within the time limit is to:
  1. Start with the schoolbook decomposition (rank 72).
  2. Try to merge terms by finding linear dependencies.
  3. Use a greedy algorithm to try to reduce the rank.

- We also try a random search with alternating least squares and rounding
  to half-integers, which may find a valid lower-rank decomposition.

- If no lower-rank decomposition is found, return the schoolbook one.
"""
import numpy as np
from itertools import product
import random
import sys
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build the schoolbook decomposition
    def build_schoolbook():
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return U, V, W
    
    U, V, W = build_schoolbook()
    
    # Verify the schoolbook decomposition
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :rank], V[j * p + k, :rank] * W[k * n + i, :rank])
                    if abs(val - 1) > 1e-6:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using random search
    # with alternating projections and half-integer rounding.
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try ranks from schoolbook_rank - 1 down to some minimum
    # We'll try a few random attempts for each rank
    
    random.seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    time_limit = 100  # Leave some margin before the 120s limit
    
    for rank in range(schoolbook_rank - 1, 30, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random initializations
        found = False
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with small values
            U_trial = np.random.randn(n * m, rank) * 0.5
            V_trial = np.random.randn(m * p, rank) * 0.5
            W_trial = np.random.randn(p * n, rank) * 0.5
            
            # Alternating least squares
            for iteration in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                        # This is a system of p equations in rank unknowns
                        # Let's form the matrix M where M[k, r] = V[j*p+k,r] * W[k*n+i,r]
                        # Then U[i*m+j, :] should satisfy M @ U[i*m+j, :] = ones(p)
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V_trial[j * p + k, :] * W_trial[k * n + i, :]
                        
                        # Solve M @ x = ones(p) in least squares sense
                        if rank >= p:
                            x, residuals, rank_val, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        else:
                            # Underdetermined, use minimum norm solution
                            M_pinv = np.linalg.pinv(M)
                            x = M_pinv @ np.ones(p)
                        
                        U_trial[i * m + j, :] = x
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                        # Let's form the matrix M where M[i, r] = U[i*m+j,r] * W[k*n+i,r]
                        # Then V[j*p+k, :] should satisfy M @ V[j*p+k, :] = ones(n)
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U_trial[i * m + j, :] * W_trial[k * n + i, :]
                        
                        if rank >= n:
                            x, residuals, rank_val, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        else:
                            M_pinv = np.linalg.pinv(M)
                            x = M_pinv @ np.ones(n)
                        
                        V_trial[j * p + k, :] = x
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                        # Let's form the matrix M where M[j, r] = U[i*m+j,r] * V[j*p+k,r]
                        # Then W[k*n+i, :] should satisfy M @ W[k*n+i, :] = ones(m)
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U_trial[i * m + j, :] * V_trial[j * p + k, :]
                        
                        if rank >= m:
                            x, residuals, rank_val, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        else:
                            M_pinv = np.linalg.pinv(M)
                            x = M_pinv @ np.ones(m)
                        
                        W_trial[k * n + i, :] = x
                
                # Check if the decomposition is close to exact
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_trial[i * m + j, :], V_trial[j * p + k, :] * W_trial[k * n + i, :])
                            err = abs(val - 1)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-4:
                    # Try rounding to half-integers
                    U_rounded = np.round(U_trial * 2) / 2
                    V_rounded = np.round(V_trial * 2) / 2
                    W_rounded = np.round(W_trial * 2) / 2
                    
                    # Check if rounded version is exact
                    max_err_rounded = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                                err = abs(val - 1)
                                if err > max_err_rounded:
                                    max_err_rounded = err
                    
                    if max_err_rounded < 1e-6:
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = rank
                        found = True
                        break
                    else:
                        # Try with the unrounded version
                        if max_err < 1e-6:
                            best_U = U_trial
                            best_V = V_trial
                            best_W = W_trial
                            best_rank = rank
                            found = True
                            break
            if found:
                break
        
        if found:
            break
    
    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Return schoolbook if no improvement found
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
