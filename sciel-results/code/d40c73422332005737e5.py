"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition as (U, V, W) lists of lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] which should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For a valid decomposition, T[i,j,k] = 1 iff (i,j,k) is a valid index, else 0.
    # Actually the tensor for matmul (A_{i,j} * B_{j,k}) -> C_{i,k}
    # The tensor T[i,j,k] = 1 for all valid i,j,k (since A_{i,j}*B_{j,k} contributes to C_{i,k})
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r]
    # The decomposition is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor form: T[i,j,k] = A_{i,j} B_{j,k} -> this is not quite right.
    # The matmul tensor is T[i,j,k] such that C_{i,k} = sum_j T[i,j,k] A_{i,j} B_{j,k}
    # Actually T[i,j,k] = 1 for all i,j,k in range, because C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T[i,j,k] = 1 for all valid indices.
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i * m + j], np.multiply(V_np[j * p + k], W_np[k * n + i]))
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using random initialization + iterative refinement
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We want rank r such that T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Also rank >= n*m*p / max(n,m,p) = 105/7 = 15 (weak)
    # Strassen-like: for 3x5x7, not directly applicable
    # Let's try ranks from 35 upward
    
    rng = random.Random(42)
    
    for rank in range(35, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random tries per rank
        max_tries = 50 if rank <= 40 else 20
        for attempt in range(max_tries):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)])
            V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)])
            W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)])
            
            # Iterative refinement: alternate between fixing two factors and solving for the third
            converged = False
            for it in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system for each (i,j)
                # Let's use least squares: for fixed i,j, we need U[i*m+j,:] such that
                # for all k: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # Let a_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product, shape rank)
                # Then sum_c U[i*m+j,c] * a_k[c] = 1 for all k
                # This is: U[i*m+j,:] @ a_k = 1 for all k
                # So A @ U[i*m+j,:]^T = 1 where A[k,c] = a_k[c]
                
                # Solve for U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # A is p x rank
                        A = np.outer(V[j * p:(j+1) * p, :], W[(0) * n + i:(0) * n + i + 0, :])  # wrong
                        # Actually: for each k, a_k[c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        # Solve A @ x = 1
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U_new[i * m + j, :] = x
                        except:
                            U_new[i * m + j, :] = U[i * m + j, :]
                U = U_new
                
                # Solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # For fixed j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let b_i = U[i*m+j,:] * W[k*n+i,:] (element-wise, shape rank)
                        # sum_c b_i[c] * V[j*p+k,c] = 1 for all i
                        B = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        try:
                            x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V_new[j * p + k, :] = x
                        except:
                            V_new[j * p + k, :] = V[j * p + k, :]
                V = V_new
                
                # Solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # For fixed k,i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                        # Let d_j = U[i*m+j,:] * V[j*p+k,:] (element-wise, shape rank)
                        # sum_c d_j[c] * W[k*n+i,c] = 1 for all j
                        D = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        try:
                            x, residuals, rank_d, sv = np.linalg.lstsq(D, np.ones(m), rcond=None)
                            W_new[k * n + i, :] = x
                        except:
                            W_new[k * n + i, :] = W[k * n + i, :]
                W = W_new
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j], np.multiply(V[j * p + k], W[k * n + i]))
                            err = max(err, abs(val - 1.0))
                if err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Verify and convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break  # Found valid decomposition at this rank, try lower? No, we're going up
    
    return best

# EVOLVE-BLOCK-END
