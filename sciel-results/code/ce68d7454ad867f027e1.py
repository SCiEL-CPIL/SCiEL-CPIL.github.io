import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j', r] * V[j'*p+k', r] * W[k*n+i, r] = delta_{j',j} * delta_{k',k}
    for all i, j, k, j', k'.
    """
    rank = len(U[0]) if U else 0
    # The condition derived:
    # For each i in [0,n), k in [0,p):
    #   S[j, jp] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r] must be delta_{j,jp}
    
    for i in range(n):
        for k in range(p):
            S = np.zeros((m, m))
            for r in range(rank):
                w_val = W[k*n+i][r]
                if w_val == 0:
                    continue
                for j in range(m):
                    u_val = U[i*m+j][r]
                    if u_val == 0:
                        continue
                    for jp in range(m):
                        v_val = V[jp*p+k][r]
                        S[j, jp] += w_val * u_val * v_val
            # Check S == I
            for j in range(m):
                for jp in range(m):
                    expected = 1 if j == jp else 0
                    if abs(S[j, jp] - expected) > 1e-9:
                        return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the max absolute error of the decomposition."""
    rank = len(U[0]) if U else 0
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            S = np.zeros((m, m))
            for r in range(rank):
                w_val = W[k*n+i][r]
                if w_val == 0:
                    continue
                for j in range(m):
                    u_val = U[i*m+j][r]
                    if u_val == 0:
                        continue
                    for jp in range(m):
                        v_val = V[jp*p+k][r]
                        S[j, jp] += w_val * u_val * v_val
            for j in range(m):
                for jp in range(m):
                    expected = 1 if j == jp else 0
                    err = abs(S[j, jp] - expected)
                    if err > max_err:
                        max_err = err
    return max_err

def _snap_to_half_int(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    arr = np.array(arr, dtype=float)
    snapped = np.round(arr * 2) / 2
    # Only snap if close
    mask = np.abs(arr - snapped) < tol
    result = np.where(mask, snapped, arr)
    return result

def _fast_optimize(U0, V0, W0, n, m, p, deadline, max_iters=500):
    """
    Optimizes U, V, W to minimize the error using gradient descent.
    Returns (U, V, W) as numpy arrays.
    """
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    rank = U.shape[1]
    
    # Precompute indices for speed
    # We need to compute S[i,k] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # This is essentially a tensor contraction.
    
    lr = 0.01
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute Error and Gradient
        err = 0.0
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for k in range(p):
                S = np.zeros((m, m))
                for r in range(rank):
                    w_val = W[k*n+i, r]
                    if w_val == 0: continue
                    for j in range(m):
                        u_val = U[i*m+j, r]
                        if u_val == 0: continue
                        for jp in range(m):
                            v_val = V[jp*p+k, r]
                            S[j, jp] += w_val * u_val * v_val
                
                # Error contribution for this i,k
                for j in range(m):
                    for jp in range(m):
                        expected = 1.0 if j == jp else 0.0
                        diff = S[j, jp] - expected
                        err += diff * diff
                        # Gradients
                        # dS[j,jp]/dW[k*n+i, r] = U[i*m+j, r] * V[jp*p+k, r]
                        # dS[j,jp]/dU[i*m+j, r] = W[k*n+i, r] * V[jp*p+k, r]
                        # dS[j,jp]/dV[jp*p+k, r] = W[k*n+i, r] * U[i*m+j, r]
                        
                        for r in range(rank):
                            w_val = W[k*n+i, r]
                            u_val = U[i*m+j, r]
                            v_val = V[jp*p+k, r]
                            
                            # Gradient for W
                            gW[k*n+i, r] += 2.0 * diff * u_val * v_val
                            # Gradient for U
                            gU[i*m+j, r] += 2.0 * diff * w_val * v_val
                            # Gradient for V
                            gV[jp*p+k, r] += 2.0 * diff * w_val * u_val
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-12:
            break
            
        # Update
        # Simple gradient step with adaptive learning rate
        gmag = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if gmag < 1e-10:
            break
        step = lr / gmag
        U -= step * gU
        V -= step * gV
        W -= step * gW
        
        # Damping if error increases
        new_err = _compute_error(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        if new_err > best_err * 1.1:
            U = best_U + step * gU # revert
            V = best_V + step * gV
            W = best_W + step * gW
            lr *= 0.5
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Schoolbook solution as baseline
    rank_school = n * m * p  # 36
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from 20 to 35
    # We know 36 works.
    # Try lower ranks.
    for target_rank in range(20, 36):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        # Use smaller scale for initialization to help convergence
        for init in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale 1.0 is a good start
            Ur = np.random.uniform(-1, 1, size=(n*m, target_rank))
            Vr = np.random.uniform(-1, 1, size=(m*p, target_rank))
            Wr = np.random.uniform(-1, 1, size=(p*n, target_rank))
            
            # Optimize
            Uo, Vo, Wo = _fast_optimize(Ur, Vr, Wr, n, m, p, deadline, max_iters=200)
            
            # Snap to half-integers
            Us = _snap_to_half_int(Uo)
            Vs = _snap_to_half_int(Vo)
            Ws = _snap_to_half_int(Wo)
            
            # Verify
            if _verify_exact(Us.tolist(), Vs.tolist(), Ws.tolist(), n, m, p):
                best_rank = target_rank
                best_U = Us
                best_V = Vs
                best_W = Ws
                found = True
                break
                
        if found:
            break
            
    # Final verification
    if not _verify_exact(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        # Fallback to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = rank_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
