"""
Strategy: Numerical optimization for tensor rank minimization.
We use the slice rank bound to find a candidate rank R.
We then attempt to find a decomposition of rank R by minimizing the Frobenius norm
of the difference between the multiplication tensor and the candidate decomposition.
We use Levenberg-Marquardt (scipy.optimize.least_squares) for this.
"""

import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def _get_lower_bound(n, m, p):
    # Slice rank bound: rank >= max(n*m, m*p, p*n)
    return max(n*m, m*p, p*n)

def _get_upper_bound(n, m, p):
    # Schoolbook upper bound
    return n*m*p

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    target_rank = _get_lower_bound(n, m, p)
    current_rank = _get_upper_bound(n, m, p)
    
    # Reshape indices for easier handling
    # U: (n*m) x R
    # V: (m*p) x R
    # W: (p*n) x R
    # Tensor T: (n*m*p) with index (i, j, k) -> value 1
    # Index mapping:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    # We want to find U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # This is a nonlinear least squares problem.
    # Variables: x = [U.flatten(), V.flatten(), W.flatten()]
    
    # Since the problem is homogeneous (scaling U by a, V by 1/a keeps product same),
    # we can fix the norm of one matrix or use a gauge fixing.
    # However, least_squares handles this reasonably if initialized well.
    
    # We will try ranks from lower_bound to upper_bound.
    # If we find a valid decomposition for rank R, we return it.
    
    # Pre-compute the target tensor flattened
    target_tensor = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in flattened tensor
                # The tensor T_{i,j,k} corresponds to the multiplication C_{i,k} += A_{i,j} B_{j,k}
                # The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}
                # We map (i,j,k) to a linear index.
                # Let's use index = i*m*p + j*p + k
                idx = i*m*p + j*p + k
                target_tensor[idx] = 1.0
    
    # Function to evaluate the error for a given set of matrices
    def residuals(x, R):
        # x is flattened [U, V, W]
        size_u = n*m*R
        size_v = m*p*R
        size_w = p*n*R
        
        U = x[:size_u].reshape(n*m, R)
        V = x[size_u:size_u+size_v].reshape(m*p, R)
        W = x[size_u+size_v:size_u+size_v+size_w].reshape(p*n, R)
        
        # Compute the tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to fill an array of shape (n, m, p)
        T = np.zeros((n, m, p))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j]
                    v_val = V[j*p+k]
                    w_val = W[k*n+i]
                    T[i,j,k] = np.dot(u_val, v_val) * 1.0 # Wait, W is also involved
                    # Correction: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is not a simple dot product of two vectors if W is distinct.
                    # It's a trilinear form.
        
        # Efficient computation:
        # For each r, we have rank-1 tensor U_r (outer of row i and j?) No.
        # U is (n*m) x R. Let U_r be the r-th column, shape (n*m).
        # V_r is the r-th column, shape (m*p).
        # W_r is the r-th column, shape (p*n).
        
        # T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # We can compute this by iterating r.
        # But for speed, let's try to vectorize.
        
        # Reshape U_r to (n, m)
        # Reshape V_r to (m, p)
        # Reshape W_r to (p, n)
        
        # T = sum_r (U_r[i,j] * V_r[j,k] * W_r[k,i])
        
        # Let's just loop over r for simplicity first, n*m*p is small (30).
        # R will be around 10-15. 30*15 = 450 ops per residual call. Fast enough.
        
        T = np.zeros(n*m*p)
        for r in range(R):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            
            # Calculate contribution of this rank-1 component
            # For each i, j, k: u_col[i,j] * v_col[j,k] * w_col[k,i]
            # We can do this via einsum or loops.
            # T_ijk += u_ijk * v_jk * w_ki
            
            # Vectorized:
            # u_expanded: (n, m, 1)
            # v_expanded: (1, m, p)
            # w_expanded: (p, 1, 1) -> need to permute to (n, m, p)
            
            # Let's use loops for i, j, k since dimensions are tiny.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m*p + j*p + k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                        
        return T - target_tensor

    # We will try to find a solution for rank R = lower_bound.
    # If it fails, we might try R+1, etc.
    # Given the small dimensions, we can try a few random initializations.
    
    best_result = None
    best_rank = current_rank
    
    # Fallback: Schoolbook
    U_sb = np.zeros((n*m, current_rank))
    V_sb = np.zeros((m*p, current_rank))
    W_sb = np.zeros((p*n, current_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    
    # Search for lower rank
    for R in range(_get_lower_bound(n, m, p), current_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with small random values
            # To avoid trivial zero solution, we might need to be careful.
            # Standard random initialization usually works for these small problems.
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R)
            
            try:
                result = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=1000)
                
                if result.success or np.linalg.norm(result.fun) < 1e-8:
                    # Check if the solution is actually valid (close to 0)
                    if np.linalg.norm(result.fun) < 1e-6:
                        # Extract matrices
                        size_u = n*m*R
                        size_v = m*p*R
                        size_w = p*n*R
                        
                        U = result.x[:size_u].reshape(n*m, R)
                        V = result.x[size_u:size_u+size_v].reshape(m*p, R)
                        W = result.x[size_u+size_v:size_u+size_v+size_w].reshape(p*n, R)
                        
                        # Verify
                        T = np.zeros(n*m*p)
                        for r in range(R):
                            u_col = U[:, r].reshape(n, m)
                            v_col = V[:, r].reshape(m, p)
                            w_col = W[:, r].reshape(p, n)
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T[i*m*p + j*p + k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                        
                        if np.allclose(T, target_tensor, atol=1e-5):
                            # We found a valid decomposition of rank R
                            best_result = {'U': U, 'V': V, 'W': W}
                            best_rank = R
                            found = True
                            break # Move to next R? Or keep searching for even lower?
                            # Since we iterate R upwards, the first found is the minimal found in this loop.
                            # But we are iterating from lower bound up. So the first success is the best.
                            # We can break out of the R loop too?
                            # The task asks for rank-minimal. If we found R, we should check if R-1 is possible?
                            # No, we started from lower bound. If lower bound is 15, we tried 15.
                            # If 15 worked, it's the best possible (since 15 is the lower bound).
                            # If 15 failed, we tried 16. If 16 worked, 16 is the best found so far.
                            # We should break the outer loop if R == lower_bound.
                            if R == _get_lower_bound(n, m, p):
                                return best_result
                            else:
                                # Keep searching for lower R? No, R is increasing.
                                # We found a valid R. We should return it?
                                # Wait, the loop is `for R in range(lower, upper)`.
                                # If R=15 fails, R=16 succeeds. We have a rank 16 solution.
                                # Can we find a rank 15 solution? We already tried.
                                # So 16 is the best we found.
                                # We can break the outer loop?
                                # Not necessarily, maybe a different random init for R=15 would have worked.
                                # But we did 10 attempts.
                                # Let's break to save time and return the best found.
                                break 
                        
            except Exception:
                pass
        
        if found:
            break # Found a valid decomposition for this R, which is lower than previous best
            
    return best_result

# Keep the interface exactly as requested
def construct():
    # The logic above is inside construct. 
    # I defined construct twice in the thought process, let's clean up.
    # The code block above defined construct. 
    # I need to make sure I only define it once.
    pass

# Re-defining construct properly to avoid confusion
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    lower_bound = _get_lower_bound(n, m, p)
    upper_bound = _get_upper_bound(n, m, p)
    
    # Fallback: Schoolbook
    U_sb = np.zeros((n*m, upper_bound))
    V_sb = np.zeros((m*p, upper_bound))
    W_sb = np.zeros((p*n, upper_bound))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = upper_bound
    
    target_tensor = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i*m*p + j*p + k] = 1.0

    def residuals(x, R):
        size_u = n*m*R
        size_v = m*p*R
        size_w = p*n*R
        
        U = x[:size_u].reshape(n*m, R)
        V = x[size_u:size_u+size_v].reshape(m*p, R)
        W = x[size_u+size_v:size_u+size_v+size_w].reshape(p*n, R)
        
        T = np.zeros(n*m*p)
        for r in range(R):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m*p + j*p + k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                        
        return T - target_tensor

    # Search for lower rank
    for R in range(lower_bound, upper_bound):
        if time.time() > deadline:
            break
            
        found = False
        # Number of attempts based on remaining time? 
        # Let's do a fixed number, say 20, to be safe.
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R)
            
            try:
                result = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=1000)
                
                # Check convergence
                if np.linalg.norm(result.fun) < 1e-6:
                    size_u = n*m*R
                    size_v = m*p*R
                    size_w = p*n*R
                    
                    U = result.x[:size_u].reshape(n*m, R)
                    V = result.x[size_u:size_u+size_v].reshape(m*p, R)
                    W = result.x[size_u+size_v:size_u+size_v+size_w].reshape(p*n, R)
                    
                    # Verify
                    T = np.zeros(n*m*p)
                    for r in range(R):
                        u_col = U[:, r].reshape(n, m)
                        v_col = V[:, r].reshape(m, p)
                        w_col = W[:, r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T[i*m*p + j*p + k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                    
                    if np.allclose(T, target_tensor, atol=1e-5):
                        best_result = {'U': U, 'V': V, 'W': W}
                        best_rank = R
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            # If we found a solution for R, and R is the lower bound, we are done.
            # If R > lower_bound, we might still try to find a lower one?
            # The loop goes from lower_bound upwards.
            # So if we found a solution for R=16, we already tried R=15.
            # We can break.
            break

    return best_result
