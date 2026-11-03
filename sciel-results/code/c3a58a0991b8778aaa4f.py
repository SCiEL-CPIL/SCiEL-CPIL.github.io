"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C from factors
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    # Efficient computation:
    # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Let S_A[i, r] = sum_j U[i*m+j, r] * A[i,j]
    # Let S_B[k, r] = sum_j V[j*p+k, r] * B[j,k]
    
    # Reshape U to (n, m, R)
    U_reshaped = U.reshape(n, m, R)
    # S_A[i, r] = sum_j U_reshaped[i, j, r] * A[i, j]
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    # Reshape V to (m, p, R) -> V[j*p+k, r] is V_flat[j, k, r] if we reshape correctly?
    # V index is j*p+k. So V is (m, p, R) if we reshape to (m, p, R)?
    # V_flat[j*p+k, r]. If we reshape V to (m, p, R), then V_reshaped[j, k, r] = V[j*p+k, r].
    V_reshaped = V.reshape(m, p, R)
    # S_B[k, r] = sum_j V_reshaped[j, k, r] * B[j, k]
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    # Reshape W to (p, n, R)
    # W index is k*n+i. So W_flat[k*n+i, r].
    # If we reshape W to (p, n, R), then W_reshaped[k, i, r] = W[k*n+i, r].
    W_reshaped = W.reshape(p, n, R)
    
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    # C_ik = sum_r W_reshaped[k, i, r] * S_A[i, r] * S_B[k, r]
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank solution
    # We will try ranks from rank_schoolbook - 1 down to rank_schoolbook - 5
    # using random restarts and least squares.
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    for R in range(rank_schoolbook - 1, max(58, rank_schoolbook - 10), -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random small values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Define the residual function
            # We want C_comp to match A @ B for several random A, B pairs?
            # Or we can define the tensor equations directly.
            # The tensor equations are:
            # For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1?
            # NO. The tensor decomposition of the multiplication tensor T is:
            # T_{i, j, k} = sum_r u_{r, i} v_{r, j} w_{r, k} ?
            # Let's re-verify the tensor definition.
            # The multiplication tensor T is a 3-tensor.
            # T_{i, j, k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # So T_{i, j, k} = 1 if the term A_{ij} B_{jk} appears in C_{ik}.
            # It appears for all i, j, k.
            # So T_{i, j, k} = 1 for all i, j, k.
            # Thus, we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
            
            # This is a system of n*m*p equations.
            # Variables: U, V, W.
            # We can flatten U, V, W into a vector x.
            # The equations are cubic (trilinear).
            
            def residuals(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute T_comp[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # We can vectorize this.
                # T_comp is (n, m, p)
                # T_comp[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                
                # Reshape U to (n, m, R)
                U_r = U_flat.reshape(n, m, R)
                V_r = V_flat.reshape(m, p, R)
                W_r = W_flat.reshape(p, n, R)
                
                # T_comp[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                # Note: W_r is (p, n, R), so W_r[k, i, r] is correct.
                # V_r is (m, p, R), so V_r[j, k, r] is correct.
                # U_r is (n, m, R), so U_r[i, j, r] is correct.
                
                # Use einsum: 'ijr,jkr,kir->ijk'
                T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                # Target is all ones
                return (T_comp - 1.0).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if res.success and res.cost < 1e-10:
                    # Verify with random matrices to be sure
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    if _verify(U_sol, V_sol, W_sol):
                        # Convert to lists for return
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                pass

    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
