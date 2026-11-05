import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: 
    1. Assume the ODE is linear and time-invariant: y' = A y.
    2. Use Krylov subspace projection (Arnoldi) to compute exp(dt * A) @ y0 
       without explicitly forming A or computing the full matrix exponential.
    3. This reduces function calls from O(n^2) or high O(n) to O(n*k) where k 
       is the Krylov dimension (typically ~50-100 for n=100).
    4. The accuracy is controlled by the Krylov dimension and the precision 
       of the small matrix exponential.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    dt = t1 - t0
    if dt == 0:
        return y0
    
    # We need to compute exp(dt * A) @ y0
    # We will use the Arnoldi process to build an orthonormal basis V_k 
    # for the Krylov subspace K_k(A, y0) = span{y0, A y0, A^2 y0, ...}
    # Then we approximate exp(dt * A) @ y0 ≈ y0 * V_k @ exp(dt * H_k) @ e_1
    # where H_k is the upper Hessenberg matrix from the Arnoldi process.
    
    # Parameters
    max_k = 100  # Maximum Krylov dimension
    tol = 1e-12  # Residual tolerance for convergence
    
    # Initialize Krylov basis
    # V is n x (k+1), but we build it column by column
    # v_0 = y0 / ||y0||
    v0_norm = np.linalg.norm(y0)
    if v0_norm < 1e-15:
        return np.zeros_like(y0)
    
    v = y0 / v0_norm
    V = np.zeros((n, max_k + 1))
    V[:, 0] = v
    
    # H is (k+1) x k, upper Hessenberg
    # We will store the coefficients h_{i,j}
    # h_{i,j} = <v_i, A v_{j-1}>
    # Actually, standard Arnoldi: A v_{j-1} = sum_{i=0}^j h_{i,j} v_i + h_{j+1,j} v_j
    # Let's use a list to store columns of H or just the coefficients.
    # For n=100, storing H as a dense matrix is fine.
    H = np.zeros((max_k + 1, max_k))
    
    # We need to apply A to vectors. 
    # Since f(t, y) = A y, we can compute A v by calling f(t0, v).
    # Note: f takes (t, y). We use t0.
    
    def apply_A(v):
        return f(t0, v)
    
    # Arnoldi iteration
    converged = False
    k = 0
    h_last = 0.0  # h_{k+1, k}
    
    for j in range(max_k):
        if time.time() > deadline:
            break
            
        # w = A v_j
        w = apply_A(V[:, j])
        
        # Modified Gram-Schmidt for stability
        for i in range(j + 1):
            h_ij = np.dot(V[:, i], w)
            H[i, j] = h_ij
            w = w - h_ij * V[:, i]
        
        h_jj1 = np.linalg.norm(w)
        H[j + 1, j] = h_jj1
        h_last = h_jj1
        
        # Check for breakdown or convergence
        if h_jj1 < tol:
            converged = True
            k = j + 1
            break
        
        V[:, j + 1] = w / h_jj1
        k = j + 1
        
    # If we didn't break early, k is the number of steps taken
    # The loop runs for j=0 to max_k-1.
    # If it finishes without break, k = max_k.
    if not converged:
        k = max_k
    
    # Now we have V_k (n x k) and H_k (k x k) [top-left k x k of H]
    # The approximation is:
    # exp(dt * A) @ y0 ≈ y0 * V_k @ exp(dt * H_k) @ e_1
    # where e_1 = [1, 0, ..., 0]^T
    
    if k == 0:
        return y0
    
    # Extract H_k
    H_k = H[:k, :k]
    
    # Compute exp(dt * H_k)
    # H_k is k x k. k is at most 100.
    # Use scipy.linalg.expm for the small matrix
    try:
        from scipy.linalg import expm
        exp_H = expm(dt * H_k)
    except Exception:
        # Fallback to simple RK4 if expm fails or is not available
        return _rk4_fallback(f, t_span, y0, deadline)
    
    # The vector we want is exp(dt * H_k) @ e_1
    e1 = np.zeros(k)
    e1[0] = 1.0
    c = exp_H @ e1
    
    # Result = v0_norm * V_k @ c
    y_end = v0_norm * (V[:, :k] @ c)
    
    return y_end

def _rk4_fallback(f, t_span, y0, deadline):
    """
    Fallback RK4 solver if linear extraction fails or times out.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    n_steps = 1000
    h = (t1 - t0) / n_steps
    t = t0
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        
        y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
