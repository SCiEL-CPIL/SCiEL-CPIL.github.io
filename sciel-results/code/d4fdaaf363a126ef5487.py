import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor is all ones for valid indices in trace convention
    # Indices: i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1
    # Flattened index for T: i * (m*p) + j * p + k ?
    # Wait, the output W index is k*n + i.
    # The tensor T is in R^(n*m) x R^(m*p) x R^(p*n).
    # Let's define the target vector t of length n*m*p.
    # t[idx] = 1 for all idx.
    
    # The decomposition computes:
    # val[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want val[i,j,k] == 1 for all i,j,k.
    
    def compute_tensor(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape to 3D for easier indexing?
        # U_3d: (n, m, R)
        # V_3d: (m, p, R)
        # W_3d: (p, n, R)
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a contraction.
        # Result shape (n, m, p)
        # Use einsum: 'ijr,jkr,kin->ij k' ? No, indices are i,j,k.
        # U index: i, j. V index: j, k. W index: k, i.
        # einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
        return np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)

    def objective(params):
        # params is flattened U, V, W
        # U size: n*m*R, V size: m*p*R, W size: p*n*R
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, R)
        V = V_flat.reshape(m * p, R)
        W = W_flat.reshape(p * n, R)
        
        T = compute_tensor(U, V, W)
        # Target is all ones
        err = np.sum((T - 1.0)**2)
        return err

    def try_rank(R, max_iterations=5000, time_limit=10):
        start_time = time.time()
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random values
            np.random.seed(attempt)
            params = np.random.randn(total_size) * 0.1
            
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': max_iterations, 'ftol': 1e-10})
            
            if res.fun < 1e-6:
                # Check for half-integer solution
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                # Snap to nearest 0.5
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                U = U_snap.reshape(n * m, R)
                V = V_snap.reshape(m * p, R)
                W = W_snap.reshape(p * n, R)
                
                T = compute_tensor(U, V, W)
                if np.allclose(T, 1.0, atol=1e-6):
                    return U, V, W
        
        return None

    # Try to find lower rank
    # Known seed rank is 12.
    # Try 10, 11.
    for R in [10, 11]:
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook rank 12
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
