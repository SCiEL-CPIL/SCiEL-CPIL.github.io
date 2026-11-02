"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 48
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition for matmul tensor."""
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        for k3 in range(p):
                            for i3 in range(n):
                                c = k3 * n + i3
                                if j == j2 and i == i3 and k2 == k3:
                                    continue
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                if abs(val) > tol:
                                    return False
        return True

    # Build the matmul tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Strategy: Try to find a lower-rank decomposition (rank 47) using optimization.
    # We attempt to fit the tensor T with r rank-1 terms.
    # If we find a solution with small residual, we have a decomposition of rank <= r.
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        # Initialize with the first r columns of the schoolbook decomposition
        # Add small random noise to break symmetry and help the optimizer
        U_init = U0[:, :r].copy()
        V_init = V0[:, :r].copy()
        W_init = W0[:, :r].copy()
        
        # Flatten for optimization
        # Variables: U (12*r), V (16*r), W (12*r)
        # Total variables: (12+16+12)*r = 40*r
        
        def residuals(params):
            U = params[:12*r].reshape(12, r)
            V = params[12*r:12*r+16*r].reshape(16, r)
            W = params[12*r+16*r:].reshape(12, r)
            
            # Compute sum of rank-1 tensors
            # T_calc[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
            # We only need to check the non-zero entries of T?
            # No, we must ensure zeros are zero too.
            # However, T is sparse. We can compute the full tensor or just the required entries.
            # Full tensor is 12x16x12 = 2304 entries.
            # Computing the full tensor in the residual function is expensive but feasible for small r.
            # Let's compute the full tensor.
            T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
            return T_calc - T
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.cost < 1e-8:
                # Extract solution
                U_sol = result.x[:12*r].reshape(12, r)
                V_sol = result.x[12*r:12*r+16*r].reshape(16, r)
                W_sol = result.x[12*r+16*r:].reshape(12, r)
                
                # Verify with strict tolerance
                if verify(U_sol, V_sol, W_sol, tol=1e-6):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        return None

    # Try rank 47
    if time.time() < deadline - 20:
        res = try_rank(rank0 - 1)
        if res is not None:
            best_rank = rank0 - 1
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")
    
    # If 47 failed, try 46? Unlikely to jump multiple ranks, but let's try 46 if time permits.
    if best_rank == rank0 and time.time() < deadline - 10:
        res = try_rank(rank0 - 2)
        if res is not None:
            best_rank = rank0 - 2
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")

    if best_rank < rank0:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
# EVOLVE-BLOCK-END
