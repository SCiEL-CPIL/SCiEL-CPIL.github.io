import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target: T[i, j, k] = 1
    # U: (n, m, R), V: (m, p, R), W: (p, n, R)
    # Equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    
    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify
        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
        if np.allclose(S, 1.0, atol=1e-9):
            return U_snap, V_snap, W_snap
        return None

    def optimize(R, seed):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n, m, R) * 0.5
        V = np.random.randn(m, p, R) * 0.5
        W = np.random.randn(p, n, R) * 0.5
        
        # Adam Optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        for step in range(5000):
            # Compute S
            S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = 1.0 - S
            
            # Gradients
            dU = -2 * np.einsum('ijk,jkr,kir->ijr', err, V, W)
            dV = -2 * np.einsum('ijk,ijr,kir->jkr', err, U, W)
            dW = -2 * np.einsum('ijk,ijr,jkr->kir', err, U, V)
            
            t += 1
            # Update U
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * (dU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            # Update V
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * (dV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            # Update W
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * (dW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            # Check convergence
            if step % 100 == 0:
                loss = np.sum(err ** 2)
                if loss < 1e-6:
                    res = snap_and_verify(U, V, W)
                    if res:
                        return res
        return None

    # Try ranks from lower bound to schoolbook
    # Lower bound for 2x3x5 is 7. Known minimal rank is likely around 10-15.
    # We will scan from 7 to 20. If not found, fallback to 30.
    
    for R in range(7, 21):
        for attempt in range(20):
            res = optimize(R, seed=R * 100 + attempt)
            if res:
                U, V, W = res
                # Convert to lists of lists matching the interface:
                # U: (n*m, rank) -> flatten (n, m, R) to (n*m, R)
                # V: (m*p, rank) -> flatten (m, p, R) to (m*p, R)
                # W: (p*n, rank) -> flatten (p, n, R) to (p*n, R)
                
                U_out = U.reshape(n * m, R).tolist()
                V_out = V.reshape(m * p, R).tolist()
                W_out = W.reshape(p * n, R).tolist()
                
                return {"U": U_out, "V": V_out, "W": W_out}
                
    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
