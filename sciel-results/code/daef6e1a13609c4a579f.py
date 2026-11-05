"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import sys

# Try to import scipy for robust optimization
try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = Fraction(1)
                V_school[j * p + k, col] = Fraction(1)
                W_school[k * n + i, col] = Fraction(1)
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    def verify_float(U_f, V_f, W_f, rank, tol=1e-8):
        """Fast float verification"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def residuals(params, R):
        """
        Compute residuals for the least squares problem.
        params: flattened vector of U, V, W
        R: rank
        """
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:size_u+size_v+size_w]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over R
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def optimize_rank_scipy(R, seed, max_time=10.0):
        """
        Try to find a rank R decomposition using scipy.optimize.least_squares
        """
        if not HAS_SCIPY:
            return None
            
        np.random.seed(seed)
        
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        total_size = size_u + size_v + size_w
        
        # Initialize
        # Scale initialization to avoid overflow/underflow
        init_val = 0.5
        params = np.random.randn(total_size) * init_val
        
        start_opt = time.time()
        
        # We might need multiple restarts if it fails
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            if attempt > 0:
                np.random.seed(seed + attempt * 100)
                params = np.random.randn(total_size) * init_val
                
            try:
                # Use trf method, which handles bounds well (though we don't use bounds here)
                # max_nfev is a good way to limit time
                result = least_squares(
                    residuals, 
                    params, 
                    args=(R,), 
                    method='trf', 
                    max_nfev=5000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                if result.cost < 1e-10:
                    # Extract solution
                    U_flat = result.x[:size_u]
                    V_flat = result.x[size_u:size_u+size_v]
                    W_flat = result.x[size_u+size_v:size_u+size_v+size_w]
                    
                    U_f = U_flat.reshape(n*m, R)
                    V_f = V_flat.reshape(m*p, R)
                    W_f = W_flat.reshape(p*n, R)
                    
                    # Verify with float
                    if verify_float(U_f, V_f, W_f, R, tol=1e-8):
                        # Convert to Fractions, rounding to nearest 0.5
                        U_rat = np.zeros((n*m, R), dtype=object)
                        V_rat = np.zeros((m*p, R), dtype=object)
                        W_rat = np.zeros((p*n, R), dtype=object)
                        
                        def to_half_int(f):
                            # Round to nearest 0.5
                            val = round(f * 2.0) / 2.0
                            return Fraction(val)
                        
                        for i in range(n*m):
                            for r in range(R):
                                U_rat[i,r] = to_half_int(U_f[i,r])
                        for i in range(m*p):
                            for r in range(R):
                                V_rat[i,r] = to_half_int(V_f[i,r])
                        for i in range(p*n):
                            for r in range(R):
                                W_rat[i,r] = to_half_int(W_f[i,r])
                        
                        # Final exact verification
                        if verify_exact(U_rat, V_rat, W_rat, R):
                            return U_rat, V_rat, W_rat, R
                            
            except Exception as e:
                pass

            # If it didn't converge, try again with different init
            if time.time() - start_opt > max_time:
                break
                
        return None

    def optimize_rank_numpy(R, seed, max_iters=2000, tol=1e-10):
        """Fallback gradient descent if scipy fails or is slow"""
        np.random.seed(seed)
        
        U_f = np.random.randn(n * m, R) * 0.5
        V_f = np.random.randn(m * p, R) * 0.5
        W_f = np.random.randn(p * n, R) * 0.5
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U_f)
        vV = np.zeros_like(V_f)
        vW = np.zeros_like(W_f)
        
        prev_err = np.inf
        plateau_count = 0
        err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute error and gradients
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] = np.sum(U_f[i*m+j, :] * V_f[j*p+k, :] * W_f[k*n+i, :])
            
            err = np.sum((T - 1.0)**2)
            
            dU = np.zeros_like(U_f)
            dV = np.zeros_like(V_f)
            dW = np.zeros_like(W_f)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = 2.0 * (T[i,j,k] - 1.0)
                        dU[i*m+j, :] += diff * V_f[j*p+k, :] * W_f[k*n+i, :]
                        dV[j*p+k, :] += diff * U_f[i*m+j, :] * W_f[k*n+i, :]
                        dW[k*n+i, :] += diff * U_f[i*m+j, :] * V_f[j*p+k, :]
            
            if err < prev_err * 0.99:
                lr = min(lr * 1.1, 0.1)
                plateau_count = 0
            else:
                plateau_count += 1
                if plateau_count > 50:
                    lr *= 0.9
                    if lr < 1e-6:
                        break
            
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U_f += vU
            V_f += vV
            W_f += vW
            
            prev_err = err
            
            if err < tol:
                break
        
        if err < tol:
            U_rat = np.zeros((n*m, R), dtype=object)
            V_rat = np.zeros((m*p, R), dtype=object)
            W_rat = np.zeros((p*n, R), dtype=object)
            
            def to_half_int(f):
                val = round(f * 2.0) / 2.0
                return Fraction(val)
                
            for i in range(n*m):
                for r in range(R):
                    U_rat[i,r] = to_half_int(U_f[i,r])
            for i in range(m*p):
                for r in range(R):
                    V_rat[i,r] = to_half_int(V_f[i,r])
            for i in range(p*n):
                for r in range(R):
                    W_rat[i,r] = to_half_int(W_f[i,r])
            
            if verify_exact(U_rat, V_rat, W_rat, R):
                return U_rat, V_rat, W_rat, R
        
        return None

    # Search for minimal rank
    best_result = None
    
    # Try Rank 25 first (theoretical minimum for this size is often cited as 25)
    if time.time() < deadline - 30:
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = optimize_rank_scipy(25, seed)
            if result:
                best_result = result
                break
            if not HAS_SCIPY:
                result = optimize_rank_numpy(25, seed)
                if result:
                    best_result = result
                    break

    # If Rank 25 failed, try Rank 26
    if not best_result and time.time() < deadline - 20:
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = optimize_rank_scipy(26, seed)
            if result:
                best_result = result
                break
            if not HAS_SCIPY:
                result = optimize_rank_numpy(26, seed)
                if result:
                    best_result = result
                    break

    # If Rank 26 failed, try Rank 27
    if not best_result and time.time() < deadline - 10:
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = optimize_rank_scipy(27, seed)
            if result:
                best_result = result
                break

    if best_result:
        U_rat, V_rat, W_rat, rank = best_result
        return {"U": [[float(x) for x in row] for row in U_rat], 
                "V": [[float(x) for x in row] for row in V_rat], 
                "W": [[float(x) for x in row] for row in W_rat]}
    else:
        # Fallback to schoolbook
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
# EVOLVE-BLOCK-END
