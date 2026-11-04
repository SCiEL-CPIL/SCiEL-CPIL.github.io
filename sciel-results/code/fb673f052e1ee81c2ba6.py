import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]
DEADLINE = 120.0 # Safety margin from 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Dimensions of the factor matrices
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Theoretical lower bound for M_{2,3,3} is 18.
    # We attempt to search for a rank < 18 decomposition, though it is known to be impossible.
    # If search fails or exceeds time, we return the standard rank 18 decomposition.
    
    def get_standard_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt optimization for rank 17
    target_rank = 17
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Random initialization
            np.random.seed(42)
            U_init = np.random.randn(dim_u, target_rank) * 0.5
            V_init = np.random.randn(dim_v, target_rank) * 0.5
            W_init = np.random.randn(dim_w, target_rank) * 0.5
            
            # Flatten for optimization
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define the tensor T that we want to match
            # T[i, j, k] = 1 if (i, j, k) is a valid multiplication term
            # Actually, the tensor T is fixed. We want sum_r U[:,r] outer V[:,r] outer W[:,r] == T
            # T is sparse. We can define the residual on the non-zero entries.
            
            # Indices of non-zero entries in T
            # T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1?
            # No, the tensor is T = sum_{i,j,k} e_{i,j}^A \otimes e_{j,k}^B \otimes e_{i,k}^C
            # The bilinear map is (A, B) -> C.
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # The tensor T has components T_{i,j,k} such that C_{i,k} = sum_{i',j',k'} T_{i',j',k'} A_{i',j'} B_{j',k'} delta_{i,i'} delta_{k,k'}?
            # No, T is in A* x B* x C.
            # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
            # So T_{i,j,k} = 1 for all i,j,k.
            # Wait, if T_{i,j,k} = 1 for all i,j,k, then the tensor is the all-ones tensor?
            # No, the indices of C are (i,k).
            # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
            # So for a fixed (i,j,k), the entry is 1.
            # There are n*m*p such entries.
            # So T is a tensor of size n x m x p, all 1s?
            # Let's check the rank of the all-ones tensor.
            # Rank of all-ones tensor of size 2x3x3 is 1.
            # But matrix multiplication is NOT the all-ones tensor.
            
            # The tensor T_{M_{n,m,p}} is defined by:
            # T_{i,j,k} = 1 if the output index (i,k) matches the input indices (i,j) and (j,k).
            # In the standard basis, the tensor is:
            # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
            # This means that for each triple (i,j,k), there is a rank-1 component.
            # The "tensor" we are decomposing is this specific structure.
            # The residual for a candidate decomposition (U, V, W) is:
            # R_{i,j,k} = (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]) - 1
            # We want R to be 0 for all i, j, k.
            
            def residual(x):
                U = x[:dim_u*target_rank].reshape(dim_u, target_rank)
                V = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                W = x[dim_u*target_rank + dim_v*target_rank :].reshape(dim_w, target_rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            res.append(val - 1.0)
                return np.array(res)
                
            # Run optimization
            result = least_squares(residual, x0, method='lm', max_nfev=1000)
            
            if result.success and result.cost < 1e-9:
                # Check if we can round to integers/half-integers
                U_opt = result.x[:dim_u*target_rank].reshape(dim_u, target_rank)
                V_opt = result.x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                W_opt = result.x[dim_u*target_rank + dim_v*target_rank :].reshape(dim_w, target_rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify rounded solution
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        except Exception:
            pass

    # Fallback to standard decomposition
    return get_standard_decomposition()
