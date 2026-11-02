import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    def check_decomp(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_approx, T, atol=1e-5):
            return False
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-5):
                return False
        return True

    def try_rank(R, max_iters=2000, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        lr = 0.01
        for it in range(max_iters):
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_approx
            
            dU = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check every 100 iterations
            if it % 100 == 0:
                if check_decomp(U, V, W):
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    return U, V, W
        return None

    # Try to find rank 20
    for seed in range(10):
        res = try_rank(20, max_iters=1000, seed=seed)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 21
    # ... (existing Strassen code)
