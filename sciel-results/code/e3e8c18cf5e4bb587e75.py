import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 21
    
    # Define the target tensor T[i, j, k]
    # T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k]
    # In the decomposition sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We need sum_r ... = 1 for the specific (k,i) derived from (i,j,k)?
    # No, the indices in W are (k,i).
    # The term A[i,j] B[j,k] contributes to C[i,k].
    # So for fixed i, j, k, the output index is (i, k).
    # Wait, W is indexed by (k, i) in the prompt: W[k*n+i].
    # So W[k, i] corresponds to C[i, k]?
    # Let's check the seed: W[k*n+i] = 1.
    # If i=0, j=0, k=0. W[0*3+0] = W[0] = 1.
    # C[0,0] is at index 0.
    # So W index (k,i) corresponds to C[i,k]?
    # If k=0, i=1. W[0*3+1] = W[1].
    # C[1,0] is at index 1.
    # So yes, W[k*n+i] corresponds to C[i,k].
    
    # So the condition is:
    # For all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For all i, j, k, k', i' with (k', i') != (k, i):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = 0
    
    # We can flatten the target tensor.
    # There are 9 input pairs (i,j) and 9 input pairs (j,k).
    # For each pair, there is 1 target output (k,i) and 8 zeros.
    # Total 81 * 9 = 729 constraints.
    
    def get_target_vector(R):
        # Returns the flattened target tensor and the mapping to variables
        # This is inefficient to build every time, but R is small.
        pass

    def residual(params, R):
        U = params[0:9*R].reshape(9, R)
        V = params[9*R:18*R].reshape(9, R)
        W = params[18*R:27*R].reshape(9, R)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target 1 at W[k, i]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res.append(val - 1)
                    
                    # Target 0 at other W[k', i']
                    for kp in range(p):
                        for ip in range(n):
                            if kp == k and ip == i:
                                continue
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[kp*n+ip, :])
                            res.append(val)
        return np.array(res)

    # Try to find a solution for R=21
    best_result = None
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        R = 21
        np.random.seed(attempt)
        x0 = np.random.randn(27 * R) * 0.1
        
        try:
            sol = least_squares(residual, x0, args=(R,), method='trf', max_nfev=1000)
        except Exception:
            continue
            
        if sol.cost < 1e-12:
            # Check if we can snap to half-integers
            x_sol = sol.x
            U = x_sol[0:9*R].reshape(9, R)
            V = x_sol[9*R:18*R].reshape(9, R)
            W = x_sol[18*R:27*R].reshape(9, R)
            
            # Snap to nearest half-integer
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify
            if np.allclose(residual(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]), R), 0, atol=1e-9):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        
        # If 21 fails, we might try 22, etc. but 21 is the known minimum.
        # If numerical search fails, we return seed.

    # Fallback to seed
    rank = n * m * p
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
