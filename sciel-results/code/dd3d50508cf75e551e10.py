"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target_tensor = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
    
    target_flat = target_tensor.flatten()
    
    def unpack_params(theta, r):
        """Unpack flattened parameter vector into U, V, W."""
        U_flat = theta[:nm * r]
        V_flat = theta[nm * r : nm * r + mp * r]
        W_flat = theta[nm * r + mp * r :]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        return U, V, W

    def residuals_full(theta, r):
        """Compute full residuals."""
        U, V, W = unpack_params(theta, r)
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        return (R - target_tensor).flatten()

    def verify(U, V, W):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        
        if np.max(np.abs(R - target_tensor)) > 1e-6:
            return False
        
        for mat in [U, V, W]:
            if not np.allclose(2.0 * mat, np.round(2.0 * mat), atol=1e-6):
                return False
        return True

    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_rank(rank, max_attempts=50, deadline=deadline):
        """Try to find a valid decomposition at the given rank."""
        param_size = rank * (nm + mp + pn)
        
        # Multiple initialization strategies
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Different initialization strategies
            if attempt % 5 == 0:
                # Random initialization
                np.random.seed(hash((rank, attempt)) % 2**32)
                theta0 = np.random.randn(param_size) * 0.3
            elif attempt % 5 == 1:
                # Sparse-like initialization with some zeros
                np.random.seed(hash((rank, attempt + 100)) % 2**32)
                theta0 = np.random.randn(param_size) * 0.5
                # Zero out some entries to encourage sparsity
                zero_mask = np.random.rand(param_size) < 0.3
                theta0[zero_mask] = 0
            elif attempt % 5 == 2:
                # Half-integer initialization
                np.random.seed(hash((rank, attempt + 200)) % 2**32)
                theta0 = np.random.randint(-2, 3, size=param_size) / 2.0
            elif attempt % 5 == 3:
                # Small random
                np.random.seed(hash((rank, attempt + 300)) % 2**32)
                theta0 = np.random.randn(param_size) * 0.1
            else:
                # Medium random
                np.random.seed(hash((rank, attempt + 400)) % 2**32)
                theta0 = np.random.randn(param_size) * 0.2
            
            try:
                res = least_squares(
                    residuals_full, 
                    theta0, 
                    args=(rank,),
                    method='trf',
                    max_nfev=500,
                    ftol=1e-12,
                    xtol=1e-12
                )
                
                if res.cost < 1e-8:
                    U, V, W = unpack_params(res.x, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = round_to_half_integer(U)
                    V_rounded = round_to_half_integer(V)
                    W_rounded = round_to_half_integer(W)
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        return (U_rounded, V_rounded, W_rounded)
                    
                    # Also try without rounding (in case exact solution is found)
                    if verify(U, V, W):
                        return (U, V, W)
                        
            except Exception:
                pass
        
        return None

    best_result = None
    best_rank = 42

    # Try ranks from 30 to 41
    for rank in range(30, 42):
        if time.time() > deadline - 10:
            break
            
        result = try_rank(rank, max_attempts=30)
        
        if result is not None:
            best_result = result
            best_rank = rank
            break

    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
