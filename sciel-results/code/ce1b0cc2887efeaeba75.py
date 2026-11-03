import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def snap_and_verify(U, V, W, tol=1e-6):
        # Snap to nearest half-integer
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        
        # Verify exactness
        # Compute tensor contraction
        # Contract U and V: sum_r U[u,r] V[v,r] -> M_uv
        # Then contract with W: sum_v M_uv W[w,v]? No.
        # Sum_r U[u,r] V[v,r] W[w,r]
        
        # Efficient contraction:
        # MV = U @ V.T  (shape nm x mp)
        # Result[u, v, w] = sum_r (U @ V.T)[u, v] ? No.
        # Let's do:
        # Temp = np.einsum('ur,vr->uv', U_h, V_h)
        # Res = np.einsum('uv,wr->uw', Temp, W_h)
        # Check if Res matches T
        
        Temp = np.einsum('ur,vr->uv', U_h, V_h)
        Res = np.einsum('uv,wr->uw', Temp, W_h)
        
        if np.allclose(Res, T, atol=tol):
            return U_h, V_h, W_h, True
        return None, None, None, False

    # Try ranks from lower bound
    min_rank = max(n * m, m * p, p * n)
    
    # Optimization function
    def optimize(rank, seed):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Gradient Descent
        lr = 0.01
        for step in range(500):
            # Compute gradients
            # L = sum (sum_r U_r V_r W_r - T)^2
            # dL/dU[u,r] = 2 * sum_{v,w} (sum_{r'} U[u,r'] V[v,r'] W[w,r'] - T[u,v,w]) * V[v,r] * W[w,r]
            
            # Vectorized gradient computation
            # Let X = T
            # Let P[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
            
            # Compute P
            # P = np.einsum('ur,vr,wr->uvw', U, V, W)
            P = np.einsum('ur,vr,wr->uvw', U, V, W)
            E = P - T
            
            # dU[u,r] = sum_{v,w} E[u,v,w] V[v,r] W[w,r]
            # dU = np.einsum('uvw,vr,wr->ur', E, V, W)
            dU = np.einsum('uvw,vr,wr->ur', E, V, W)
            
            # dV[v,r] = sum_{u,w} E[u,v,w] U[u,r] W[w,r]
            # dV = np.einsum('uvw,ur,wr->vr', E, U, W)
            dV = np.einsum('uvw,ur,wr->vr', E, U, W)
            
            # dW[w,r] = sum_{u,v} E[u,v,w] U[u,r] V[v,r]
            # dW = np.einsum('uvw,ur,vr->wr', E, U, V)
            dW = np.einsum('uvw,ur,vr->wr', E, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic snap check
            if step % 50 == 49:
                U_h, V_h, W_h, ok = snap_and_verify(U, V, W)
                if ok:
                    return U_h, V_h, W_h, True
        
        # Final check
        U_h, V_h, W_h, ok = snap_and_verify(U, V, W)
        if ok:
            return U_h, V_h, W_h, True
        
        return None, None, None, False

    # Try to find a solution with rank = min_rank
    for rank in range(min_rank, min_rank + 10):
        for seed in range(5):
            U, V, W, ok = optimize(rank, seed + rank * 100)
            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1.0
                V[v, col] = 1.0
                W[w, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
