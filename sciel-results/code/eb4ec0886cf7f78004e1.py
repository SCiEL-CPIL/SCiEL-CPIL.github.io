"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    # Compute sum_c U[:,c] * V[:,c] * W[:,c]
    # This is a 3-tensor: result[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # Efficient: for each c, outer product
    result = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol)

def _try_rank(r, n, m, p, max_iters=200, seed=42):
    """Try to find a rank-r decomposition using iterative refinement."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    for attempt in range(max_iters):
        # Initialize with random half-integer values
        U = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(n*m, r))
        V = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(m*p, r))
        W = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(p*n, r))
        
        # Iterate: fix two, solve for third
        for it in range(50):
            # Solve for U given V, W
            # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
            # For fixed j,k: T[:,j,k] = U * diag(V[j,:]*W[k,:])
            # This is a linear system
            for c in range(r):
                pass  # Simplified: just check convergence
            
            # Check if close to solution
            result = np.zeros((n*m, m*p, p*n))
            for c in range(r):
                result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            err = np.linalg.norm(result - T)
            if err < 1e-9:
                # Round to half-integers and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if _validate(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        # If not converged, try again
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best = (rank_school, U, V, W)
    
    # Try to find lower rank
    for r in range(16, rank_school):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iters=50, seed=r)
        if result is not None:
            U_r, V_r, W_r = result
            best = (r, U_r, V_r, W_r)
            break  # Found minimal, stop
    
    _, U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
