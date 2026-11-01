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
    deadline = 140.0  # safety margin before 150s limit
    
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
    
    def unpack_params(theta, r):
        """Unpack flattened parameter vector into U, V, W."""
        U_flat = theta[:nm * r]
        V_flat = theta[nm * r : nm * r + mp * r]
        W_flat = theta[nm * r + mp * r :]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        return U, V, W

    def compute_residuals(U, V, W, target_tensor):
        """Compute residuals efficiently."""
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        return (R - target_tensor).flatten()

    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        
        if np.max(np.abs(R - target_tensor)) > tol:
            return False
        
        for mat in [U, V, W]:
            if not np.allclose(2.0 * mat, np.round(2.0 * mat), atol=tol):
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

    def structured_init(rank, seed):
        """Create structured initial guess based on matrix multiplication structure."""
        np.random.seed(seed)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Initialize based on the structure of matrix multiplication
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    
                    # Assign each (i,j,k) triple to a random column
                    col = np.random.randint(0, rank)
                    weight = np.random.choice([1, -1, 0.5, -0.5])
                    
                    U[a, col] += weight
                    V[b, col] += weight
                    W[c, col] += weight
        
        # Add some random noise to break symmetries
        U += np.random.randn(nm, rank) * 0.1
        V += np.random.randn(mp, rank) * 0.1
        W += np.random.randn(pn, rank) * 0.1
        
        return U, V, W

    def random_init(rank, seed, scale=0.3):
        """Random initialization."""
        np.random.seed(seed)
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        return U, V, W

    def try_rank(rank, max_attempts=20, deadline=deadline):
        """Try to find a valid decomposition at the given rank."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Different initialization strategies
            if attempt % 3 == 0:
                U, V, W = structured_init(rank, hash((rank, attempt)) % 2**32)
            elif attempt % 3 == 1:
                U, V, W = random_init(rank, hash((rank, attempt + 100)) % 2**32, scale=0.2)
            else:
                U, V, W = random_init(rank, hash((rank, attempt + 200)) % 2**32, scale=0.5)
            
            theta0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(theta, r=rank):
                U_local, V_local, W_local = unpack_params(theta, r)
                return compute_residuals(U_local, V_local, W_local, target_tensor)
            
            try:
                res = least_squares(
                    residuals, 
                    theta0,
                    method='trf',
                    max_nfev=200,
                    ftol=1e-10,
                    xtol=1e-10
                )
                
                if res.cost < 1e-6:
                    U_opt, V_opt, W_opt = unpack_params(res.x, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = round_to_half_integer(U_opt)
                    V_rounded = round_to_half_integer(V_opt)
                    W_rounded = round_to_half_integer(W_opt)
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        return (U_rounded, V_rounded, W_rounded)
                    
                    # Try without rounding
                    if verify(U_opt, V_opt, W_opt):
                        return (U_opt, V_opt, W_opt)
                        
                    # Try a final polish with higher precision
                    if res.cost < 1e-4:
                        def residuals_highprec(theta, r=rank):
                            U_local, V_local, W_local = unpack_params(theta, r)
                            return compute_residuals(U_local, V_local, W_local, target_tensor)
                        
                        try:
                            res2 = least_squares(
                                residuals_highprec,
                                res.x,
                                method='trf',
                                max_nfev=100,
                                ftol=1e-14,
                                xtol=1e-14
                            )
                            U_polished, V_polished, W_polished = unpack_params(res2.x, rank)
                            U_polished_r = round_to_half_integer(U_polished)
                            V_polished_r = round_to_half_integer(V_polished)
                            W_polished_r = round_to_half_integer(W_polished)
                            
                            if verify(U_polished_r, V_polished_r, W_polished_r):
                                return (U_polished_r, V_polished_r, W_polished_r)
                            if verify(U_polished, V_polished, W_polished):
                                return (U_polished, V_polished, W_polished)
                        except Exception:
                            pass
                        
            except Exception:
                pass
        
        return None

    best_result = None
    best_rank = 42

    # Try ranks from 25 to 41
    for rank in range(25, 42):
        if time.time() > deadline - 5:
            break
            
        result = try_rank(rank, max_attempts=15)
        
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
