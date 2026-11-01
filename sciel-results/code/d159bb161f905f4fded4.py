import numpy as np
import time
import random

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    # Target rank: 
    # Theoretical lower bound is often higher than max(nm, mp, pn).
    # For 2x2 * 2x9:
    # Schoolbook rank = 36.
    # Strassen-like reductions might get us to ~20-24.
    # Let's try to find a rank R decomposition.
    
    # We will attempt to find a decomposition for ranks from 18 up to 35.
    # If we find a valid one, we return it.
    # If we run out of time or fail, we return the schoolbook rank 36.

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}

    # The tensor T for matrix multiplication M_{n,m} * M_{m,p}
    # T[i, j, k] = 1 for all i, j, k? 
    # No, the tensor definition is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k} T_{i,j,k}.
    # So T is the all-ones tensor of shape (n, m, p).
    # We want to decompose T into sum_r U_r * V_r * W_r (outer products).
    # Specifically: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def try_rank(R):
        # Initialize U, V, W with random values
        # Using small random values to avoid explosion
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # We want to minimize || T - sum_r U_r V_r W_r ||^2
        # This is equivalent to minimizing the Frobenius norm of the difference.
        
        # Gradient Descent
        # Let D_r = V_r * W_r (element-wise? No, V_r is vector, W_r is vector. 
        # The term is U_r (outer) V_r (outer) W_r.
        # Let's flatten the target tensor T to a vector t of length n*m*p.
        # The model is sum_r (U[:,r] * V[:,r] * W[:,r]) where * is element-wise?
        # No. 
        # Index mapping:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # This coupling makes it hard to vectorize simply as element-wise product of flattened vectors.
        
        # Let's construct the full tensor difference explicitly for small dimensions.
        # n=2, m=2, p=9. Total size 36.
        # We can represent the current decomposition tensor C as a 3D array (n, m, p).
        
        # C[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can compute gradients analytically.
        # Let E = T - C.
        # dL/dU[i*m+j, r] = sum_{k} E[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
        # dL/dV[j*p+k, r] = sum_{i} E[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
        # dL/dW[k*n+i, r] = sum_{j} E[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
        
        # We will use a simple gradient descent with momentum or just fixed step.
        # Since dimensions are tiny, we can afford to compute C explicitly each step.
        
        # Precompute index maps for speed?
        # i in 0..n-1, j in 0..m-1, k in 0..p-1
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        
        # Let's iterate steps.
        steps = 2000
        lr = 0.05
        momentum = 0.9
        
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        # Target tensor T is all ones.
        # We can just compute C and error.
        
        # To speed up, we can precompute the indices for each r?
        # Actually, n*m*p is 36. R is ~20.
        # Computing C: 
        # C = np.einsum('ir,jr,kr->ijk', U, V, W) ? 
        # No, the indices are coupled.
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
        
        # Let's unroll the loops for the specific dimensions to be fast?
        # Or just use numpy broadcasting carefully.
        
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # C[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        # This is a contraction over r.
        
        for step in range(steps):
            if time.time() > deadline - 10.0:
                return None
            
            # Compute C
            # C[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
            # We can do this by iterating r? Or using einsum?
            # einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            C = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            # Error E = T - C. T is all ones.
            E = 1.0 - C
            
            # Compute gradients
            # dU[i, j, r] = sum_k E[i, j, k] * V_3d[j, k, r] * W_3d[k, i, r]
            # dV[j, k, r] = sum_i E[i, j, k] * U_3d[i, j, r] * W_3d[k, i, r]
            # dW[k, i, r] = sum_j E[i, j, k] * U_3d[i, j, r] * V_3d[j, k, r]
            
            # dU = np.einsum('ijk,jkr,kir->ijr', E, V_3d, W_3d)
            # dV = np.einsum('ijk,ijr,kir->jkr', E, U_3d, W_3d)
            # dW = np.einsum('ijk,ijr,jkr->kir', E, U_3d, V_3d)
            
            dU = np.einsum('ijk,jkr,kir->ijr', E, V_3d, W_3d)
            dV = np.einsum('ijk,ijr,kir->jkr', E, U_3d, W_3d)
            dW = np.einsum('ijk,ijr,jkr->kir', E, U_3d, V_3d)
            
            # Update with momentum
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U_3d += vU
            V_3d += vV
            W_3d += vW
            
            # Check convergence
            # If max abs error is small, we are good.
            # But we need exact decomposition? 
            # "integer/half-integer entries that exactly decompose"
            # Numerical methods find real approximations. 
            # We must round to nearest half-integer and verify.
            
            if step % 50 == 0:
                max_err = np.max(np.abs(E))
                if max_err < 1e-6:
                    # Try to round
                    U_rounded = np.round(U_3d.flatten() * 2) / 2.0
                    V_rounded = np.round(V_3d.flatten() * 2) / 2.0
                    W_rounded = np.round(W_3d.flatten() * 2) / 2.0
                    
                    # Verify rounded version
                    Ur = U_rounded.reshape(n, m, R)
                    Vr = V_rounded.reshape(m, p, R)
                    Wr = W_rounded.reshape(p, n, R)
                    Cr = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
                    if np.allclose(Cr, 1.0, atol=1e-9):
                        return {
                            "U": U_rounded.reshape(-1, R).tolist(),
                            "V": V_rounded.reshape(-1, R).tolist(),
                            "W": W_rounded.reshape(-1, R).tolist()
                        }
        
        return None

    # Try ranks from 18 upwards
    # Note: Rank 18 is a guess. Theoretical min rank for 2x2*2x2 is 7.
    # For 2x2*2x9, we can view it as 3 blocks of 2x2*2x3? No, 9 is not multiple of 2.
    # 2x2*2x9 is equivalent to 2x2*2x2 + 2x2*2x2 + 2x2*2x2 + 2x2*2x1.
    # Rank(2x2*2x2) <= 7. Rank(2x2*2x1) <= 4.
    # So Rank <= 7*3 + 4 = 25.
    # Let's try 25, 24, 23, 22, 21, 20, 19, 18.
    
    best_result = None
    for R in range(18, 36):
        if time.time() > deadline - 20.0:
            break
            
        # Multiple restarts for each rank to escape local minima
        found = False
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            result = try_rank(R)
            if result is not None:
                best_result = result
                found = True
                break
        if found:
            break
            
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()

