"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            # Compute C[i,k] from decomposition
            left = np.zeros((m, m))
            for c in range(rank):
                u_i = U[i*m:(i+1)*m, c]
                v_k = V[:, c]
                # u_i is indexed by j, v_k is indexed by (j', k)
                # We need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
                for j in range(m):
                    for jp in range(m):
                        left[j, jp] += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
            # left should be identity
            if not np.allclose(left, np.eye(m), atol=tol):
                return False
    return True

def _try_rank(n, m, p, target_rank, deadline):
    """Try to find a decomposition of given rank."""
    best = None
    np.random.seed(42)
    max_iters = 20000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Random initialization
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Try to refine: we need sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}
        # This is a system of equations. Let's use iterative updates.
        converged = False
        for inner in range(200):
            if time.time() > deadline:
                break
            # Update U
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            # Target: delta_{j,jp}
                            target = 1.0 if j == jp else 0.0
                            # Current contribution from other terms
                            # This is complex; use a simple gradient-like step
                            pass
            # Check convergence
            if _verify(U, V, W, n, m, p, tol=1e-6):
                converged = True
                break
        if converged:
            # Polish
            for polish in range(100):
                if time.time() > deadline:
                    break
                if _verify(U, V, W, n, m, p, tol=1e-10):
                    break
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try lower ranks first
    for target_rank in range(17, 17, -1):  # Just 17 for now
        result = _try_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            # Convert to lists with integer/half-integer if possible
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
