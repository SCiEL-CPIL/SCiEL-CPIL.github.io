import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    # Try to find a rank-14 solution using a structured search
    # R(2,3,3) = 14 is known from tensor decomposition literature
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition using Fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j, r]) * Fraction(V[j * p + k, r]) * Fraction(W[k * n + i, r])
                    if total != 1:
                        return False
        return True
    
    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)
    
    # Try rank 14 with a structured initialization approach
    # Use a smarter initialization based on the structure of matrix multiplication
    
    for target_rank in [14, 15, 16, 17]:
        if time.time() - start_time > deadline:
            break
        
        found = False
        max_restarts = 60 if target_rank == 14 else 30
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            
            rng = np.random.RandomState(restart * 7919 + target_rank * 104729)
            
            # Initialize with structured random values
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a mix of structured patterns and random initialization
            for r in range(target_rank):
                # Pattern-based initialization
                if r < n * m:
                    i, j = divmod(r, m)
                    U[i * m + j, r] = 1.0
                if r < m * p:
                    j, k = divmod(r, p)
                    V[j * p + k, r] = 1.0
                if r < p * n:
                    k, i = divmod(r, n)
                    W[k * n + i, r] = 1.0
            
            # Add random perturbation
            U += rng.randn(n * m, target_rank) * 0.5
            V += rng.randn(m * p, target_rank) * 0.5
            W += rng.randn(p * n, target_rank) * 0.5
            
            # ALS iterations with early stopping
            prev_error = np.inf
            for iter_ in range(100):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((target_rank, p))
                        for k in range(p):
                            A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, T[i, j, :].reshape(-1), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((target_rank, n))
                        for i in range(n):
                            A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, T[:, j, k].reshape(-1), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((target_rank, m))
                        for j in range(m):
                            A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, T[i, :, k].reshape(-1), rcond=None)[0]
                        except:
                            pass
                
                error = compute_error(U, V, W, target_rank)
                
                if error < 1e-10:
                    break
                
                if error > prev_error * 1.01 and iter_ > 10:
                    break
                prev_error = error
            
            # Check if we found a good solution
            if prev_error < 1e-8:
                # Round to nearest half-integer and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # If no lower-rank solution found, use schoolbook
    if best_rank == R_school:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
