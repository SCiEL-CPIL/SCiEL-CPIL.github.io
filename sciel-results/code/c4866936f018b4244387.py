import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build the target tensor T[i,j,k] = delta_{i,k} (for matrix multiplication)
    # T[i,j,k] = 1 if i==k, else 0, where i is row of A, j is col of A / row of B, k is col of B
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = delta_{i,k}... no wait.
    # The multiplication tensor has T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the "identity" in the middle index)
    # Actually: T[i,j,k] = 1 if j is the shared index, which it always is.
    # The tensor is: T[i,j,k] = 1 for all i,j,k in their respective ranges
    # No wait - the multiplication tensor is defined such that
    # sum_r U[i,r] V[j,r] W[k,r] = delta_{j,j} ... 
    # Let me think again. For matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So we need: sum_r U[i,j,r] V[j,k,r] W[i,k,r] = delta_{j,j} = 1? No.
    # We need: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # That's not right either. The tensor has entries T[i,j,k] where:
    # - i indexes rows of A (and C)
    # - j indexes cols of A (and rows of B)  
    # - k indexes cols of B (and C)
    # And T[i,j,k] = 1 if this combination contributes to C[i,k], which is always true.
    # So T is all ones? No, that's the trivial upper bound.
    
    # Correct definition: The matrix multiplication tensor M_{n,m,p} has
    # M_{i,j,k} = 1 if i==k (no, that's not right)
    # Actually: M_{i,j,k} = 1 for all i,j,k. The "tensor" is just the all-ones tensor
    # in the sense that every (i,j,k) triple has a 1.
    # Wait no. Let me be precise.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the bilinear form has terms A[i,j] * B[j,k] with coefficient 1 for each (i,j,k).
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # But that's n*m*p = 4*5*5 = 100 terms in schoolbook.
    
    # For the decomposition: sum_r U[i,r] V[j,r] W[k,r] should equal T[i,j,k] = 1 for all i,j,k
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # and U[i*m+j, r] is the coefficient for A[i,j], etc.
    
    # So we need: for all i in [0,4), j in [0,5), k in [0,5):
    #   sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*4+i, r] = 1
    
    # Let's use numerical optimization to find a low-rank decomposition
    start_time = time.time()
    deadline = start_time + 120.0  # Leave margin for safety
    
    # Start with schoolbook as initial solution
    U_init = np.zeros((n_m, n*m*p))
    V_init = np.zeros((m_p, n*m*p))
    W_init = np.zeros((p_n, n*m*p))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i*m+j, idx] = 1.0
                V_init[j*p+k, idx] = 1.0
                W_init[k*n+i, idx] = 1.0
                idx += 1
    
    # Try to reduce rank using ALS-style optimization
    # We'll try different target ranks and see which ones work
    
    best_U, best_V, best_W = None, None, None
    best_rank = n*m*p  # 100
    
    # Try various target ranks
    target_ranks = [80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]
    
    for target_r in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Initialize with random projections of the schoolbook solution
        rng = np.random.RandomState(42)
        
        # Use truncated SVD-like approach
        # Project U, V, W onto rank-r subspaces
        U_r = np.zeros((n_m, target_r))
        V_r = np.zeros((m_p, target_r))
        W_r = np.zeros((p_n, target_r))
        
        # Initialize by taking first r columns and doing ALS
        U_r[:, :min(target_r, n*m*p)] = U_init[:, :min(target_r, n*m*p)]
        V_r[:, :min(target_r, n*m*p)] = V_init[:, :min(target_r, n*m*p)]
        W_r[:, :min(target_r, n*m*p)] = W_init[:, :min(target_r, n*m*p)]
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each i, solve least squares
            # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # For fixed i: sum_r U[i,r] (V[j,r] W[k,r]) = 1 for all j,k
            # Let M_VW[j,k] = sum_r V[j,r] W[k,r]
            # Then we need U[i,:] @ M_VW = 1 for all j,k
            # This is: for each i, U[i,:] is a vector such that U[i,:] @ M_VW = ones(m*p)
            
            M_VW = V_r @ W_r.T  # shape (m*p, p*n)
            # We need U[i*m+j, :] such that sum over r: U[i*m+j,r] * M_VW[j*p+k, k*n+i] = 1
            # This is tricky. Let me reformulate.
            
            # Actually, let's do proper ALS:
            # Fix V and W, solve for U:
            # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # Let A_r = V[j*p+k, r] * W[k*n+i, r] for each k
            # Then sum_r U[i*m+j, r] * A_r = 1
            # This is a linear system in U[i*m+j, :]
            
            # Build the system: for fixed (i,j), we have p equations (one for each k)
            # and target_r unknowns (U[i*m+j, 0..target_r-1])
            # If target_r > p, we have an underdetermined system.
            # Use least squares.
            
            for i in range(n):
                for j in range(m):
                    # Build matrix B where B[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    B = np.zeros((p, target_r))
                    for k in range(p):
                        B[k, :] = V_r[j*p+k, :] * W_r[k*n+i, :]
                    # Solve B @ u = ones(p)
                    if target_r <= p:
                        u, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    else:
                        # Underdetermined: find minimum norm solution
                        # Use pseudoinverse
                        u = np.linalg.pinv(B) @ np.ones(p)
                    U_r[i*m+j, :] = u
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_r))
                    for i in range(n):
                        B[i, :] = U_r[i*m+j, :] * W_r[k*n+i, :]
                    if target_r <= n:
                        v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    else:
                        v = np.linalg.pinv(B) @ np.ones(n)
                    V_r[j*p+k, :] = v
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, target_r))
                    for j in range(m):
                        B[j, :] = U_r[i*m+j, :] * V_r[j*p+k, :]
                    if target_r <= m:
                        w, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    else:
                        w = np.linalg.pinv(B) @ np.ones(m)
                    W_r[k*n+i, :] = w
            
            # Check convergence
            if it % 10 == 9:
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(target_r))
                            err += (val - 1.0)**2
                if err < 1e-8:
                    converged = True
                    break
        
        if converged:
            # Verify the solution
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(target_r))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                best_rank = target_r
                break  # Found a valid solution, try smaller ranks
    
    # If we found a solution, use it; otherwise fall back to schoolbook
    if best_U is not None:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = U_init, V_init, W_init
    
    # Round to half-integers if possible
    def round_to_half_int(arr):
        # Try to round each entry to nearest half-integer
        rounded = np.round(arr * 2) / 2
        # Check if rounding preserves the decomposition
        return rounded
    
    # Verify the final solution
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(U.shape[1]))
                    if abs(val - 1.0) > 1e-4:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fall back to schoolbook
        U, V, W = U_init, V_init, W_init
    
    # Remove zero columns
    nonzero_cols = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    U = U[:, nonzero_cols]
    V = V[:, nonzero_cols]
    W = W[:, nonzero_cols]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
