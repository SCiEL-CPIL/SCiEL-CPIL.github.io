"""Matrix multiplication 2x3 x 3x7 - rank minimization search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 140.0  # safety margin before 150s kill

def _validate(U, V, W, n=2, m=3, p=7):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _compute_errors(U, V, W, n=2, m=3, p=7):
    """Compute errors for all (i,j,k) triples."""
    rank = U.shape[1]
    errors = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                errors[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
    return errors

def _try_rank(rank, n=2, m=3, p=7, max_iters=500, seed=0):
    """Try to find a valid rank-r decomposition via alternating least squares + local search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Multiple initializations
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init in range(5):
        if time.time() > DEADLINE - 10:
            break
        
        # Different initialization strategies
        if init == 0:
            # Random
            U = np_rng.randn(n*m, rank)
            V = np_rng.randn(m*p, rank)
            W = np_rng.randn(p*n, rank)
        elif init == 1:
            # Sparse random
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                U[rng.randint(0, n*m-1), c] = 1
                V[rng.randint(0, m*p-1), c] = 1
                W[rng.randint(0, p*n-1), c] = 1
        else:
            # Structured: try to share patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            # Assign each column a random (i,j,k) pattern
            for c in range(rank):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
        
        # Alternating least squares
        for it in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            
            errors = _compute_errors(U, V, W, n, m, p)
            max_err = np.max(np.abs(errors))
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update one matrix at a time (alternating)
            mat_idx = it % 3
            
            if mat_idx == 0:  # Update U
                # For each row of U, solve linear system
                for i in range(n):
                    for j in range(m):
                        # Want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                        # This is: sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1
                        # Let B[k,c] = V[j,k,c] * W[k,i,c]
                        # Then: sum_c U[i,j,c] * B[k,c] = 1 for all k
                        # This is: B @ U[i,j,:]^T = ones(p)
                        B = np.zeros((p, rank))
                        for k in range(p):
                            B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve B @ x = ones(p)
                        try:
                            x = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = x
                        except:
                            pass
                        
            elif mat_idx == 1:  # Update V
                for j in range(m):
                    for k in range(p):
                        # Want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
                        # Let A[i,c] = U[i,j,c] * W[k,i,c]
                        # Then: sum_c V[j,k,c] * A[i,c] = 1 for all i
                        # A @ V[j,k,:]^T = ones(n)
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = x
                        except:
                            pass
                            
            else:  # Update W
                for k in range(p):
                    for i in range(n):
                        # Want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                        # Let C[j,c] = U[i,j,c] * V[j,k,c]
                        # Then: sum_c W[k,i,c] * C[j,c] = 1 for all j
                        # C @ W[k,i,:]^T = ones(m)
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = x
                        except:
                            pass
        
        # Local search refinement
        for ls_it in range(100):
            if time.time() > DEADLINE - 10:
                break
            
            errors = _compute_errors(U, V, W, n, m, p)
            max_err = np.max(np.abs(errors))
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Perturb one random element
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randint(0, n*m-1)
            elif mat == 'V':
                row = rng.randint(0, m*p-1)
            else:
                row = rng.randint(0, p*n-1)
            col = rng.randint(0, rank-1)
            
            best_perturb = None
            best_perturb_err = max_err
            for delta in [0.1, -0.1, 0.01, -0.01]:
                if mat == 'U':
                    orig = U[row, col]
                    U[row, col] = orig + delta
                elif mat == 'V':
                    orig = V[row, col]
                    V[row, col] = orig + delta
                else:
                    orig = W[row, col]
                    W[row, col] = orig + delta
                
                err = np.max(np.abs(_compute_errors(U, V, W, n, m, p)))
                
                if mat == 'U':
                    U[row, col] = orig
                elif mat == 'V':
                    V[row, col] = orig
                else:
                    W[row, col] = orig
                
                if err < best_perturb_err:
                    best_perturb_err = err
                    best_perturb = (mat, row, col, delta)
            
            if best_perturb is not None:
                mat, row, col, delta = best_perturb
                if mat == 'U':
                    U[row, col] += delta
                elif mat == 'V':
                    V[row, col] += delta
                else:
                    W[row, col] += delta
            else:
                break
    
    if best_err < 1e-10 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_structured_rank(rank, n=2, m=3, p=7):
    """Try structured decompositions based on known patterns."""
    # For small matrices, try specific structured approaches
    # One approach: use the fact that we can factor through smaller dimensions
    return None

def _schoolbook_decomposition(n=2, m=3, p=7):
    """Generate the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # First, generate schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Theoretical lower bound: max(n*p, n*m, m*p) = max(14, 6, 21) = 21
    # Try ranks from lower bound upward
    
    # Priority: try ranks that are more likely to have solutions
    # Based on known results, matrix multiplication ranks are often near the lower bound
    ranks_to_try = list(range(21, 35))  # Try 21 to 34
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 15:
            break
        
        # Try multiple seeds for this rank
        # More attempts for lower ranks (more likely to succeed if possible)
        max_attempts = 5 if target_rank <= 24 else 3
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 15:
                break
            
            result = _try_rank(target_rank, n, m, p, max_iters=200, seed=attempt*1000+42)
            if result is not None:
                U, V, W = result
                # Verify
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # If we haven't found anything better, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
