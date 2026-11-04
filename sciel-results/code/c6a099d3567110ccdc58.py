"""Matrix multiplication 3x4x5 - rank-minimal tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build the target tensor: T[i,j,k] = 1 for all (i,j,k) in the multiplication
    # Actually the multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = delta_{j,j} which is 1 for all valid i,j,k
    # But the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank(rank, n, m, p, rng, max_iter=2000, tol=1e-10):
    """Try to find a decomposition of given rank using optimization."""
    # Initialize with random values
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))
    
    # Scale to avoid numerical issues
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    # We want: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a non-convex problem. Use alternating updates.
    # For fixed V,W, U[i*m+j,r] should satisfy the linear constraints.
    # For fixed U,W, V[j*p+k,r] should satisfy linear constraints.
    # For fixed U,V, W[k*n+i,r] should satisfy linear constraints.
    
    # Let's use a gradient-based approach with the objective:
    # sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    lr = 0.01
    for it in range(max_iter):
        # Compute gradients
        # dL/dU[i*m+j,r] = sum_k 2*(pred[i,j,k]-1)*V[j*p+k,r]*W[k*n+i,r]
        # This is expensive for large rank. Let's use a simpler alternating scheme.
        
        # Alternating least squares approach:
        # Fix V and W, solve for U (linear)
        # Fix U and W, solve for V (linear)
        # Fix U and V, solve for W (linear)
        
        # For U: for each (i,j), we have equations:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is: U[i*m+j,:] @ (V[j*p,:]*W[i,:].T reshaped) ... 
        # Actually: for fixed i,j, the equation over k is:
        # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for each k
        # Let M_k(r) = V[j*p+k,r] * W[k*n+i,r], then sum_r U[i*m+j,r]*M_k(r) = 1
        # This is a linear system in U[i*m+j,:].
        
        # Similarly for V and W.
        
        # Let's just do simple gradient descent with small steps
        # Compute predictions
        pred = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
        
        err = pred - 1.0
        if np.max(np.abs(err)) < tol:
            return U, V, W, True
        
        # Gradient descent
        # dU[i*m+j,r] = sum_k 2*err[i,j,k]*V[j*p+k,r]*W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                grad_U = np.zeros(rank)
                for k in range(p):
                    grad_U += 2 * err[i,j,k] * V[j*p+k,:] * W[k*n+i,:]
                U[i*m+j,:] -= lr * grad_U
        
        for j in range(m):
            for k in range(p):
                grad_V = np.zeros(rank)
                for i in range(n):
                    grad_V += 2 * err[i,j,k] * U[i*m+j,:] * W[k*n+i,:]
                V[j*p+k,:] -= lr * grad_V
        
        for k in range(p):
            for i in range(n):
                grad_W = np.zeros(rank)
                for j in range(m):
                    grad_W += 2 * err[i,j,k] * U[i*m+j,:] * V[j*p+k,:]
                W[k*n+i,:] -= lr * grad_W
    
    # Final check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return U, V, W, max_err < tol

def _snap_to_rational(U, V, W, denom=2):
    """Try to snap values to multiples of 1/denom."""
    U_s = np.round(U * denom) / denom
    V_s = np.round(V * denom) / denom
    W_s = np.round(W * denom) / denom
    return U_s, V_s, W_s

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Start with schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower ranks
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12,20,15)=20
    # Also rank >= n+m+p-2 = 10 (for matrix mult, this isn't tight)
    # Let's try ranks from 30 upward (reasonable starting point)
    
    rng = np.random.default_rng(42)
    
    for rank in range(30, rank_sb):
        if time.time() - start_time > deadline:
            break
        
        # Multiple random restarts
        found = False
        for restart in range(5):
            if time.time() - start_time > deadline:
                break
            
            U, V, W, success = _try_rank(rank, n, m, p, rng, max_iter=500, tol=1e-8)
            
            if success:
                # Try to snap to rationals
                U_s, V_s, W_s = _snap_to_rational(U, V, W, denom=2)
                # Verify snapped version
                U_arr = np.array(U_s)
                V_arr = np.array(V_s)
                W_arr = np.array(W_s)
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    # Convert to lists
                    best_rank = rank
                    best_U = U_s.tolist()
                    best_V = V_s.tolist()
                    best_W = W_s.tolist()
                    found = True
                    break
                else:
                    # Use the non-snapped version if it's very close
                    if max_err < 1e-6:
                        best_rank = rank
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                        found = True
                        break
        
        if found:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_sb:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
