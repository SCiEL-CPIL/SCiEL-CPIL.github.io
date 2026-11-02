import numpy as np
import time
import scipy.optimize

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    def validate(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    if j*p <= b < (j+1)*p:
                        k = b - j*p
                        if c == k*n + i:
                            continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def get_tensor_indices():
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    indices.append((a, b, c, 1.0))
        return indices

    def get_zero_indices():
        indices = []
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    if j*p <= b < (j+1)*p:
                        k = b - j*p
                        if c == k*n + i:
                            continue
                    indices.append((a, b, c, 0.0))
        return indices

    target_indices = get_tensor_indices()
    zero_indices = get_zero_indices()

    def objective_and_grad(x):
        r = x.shape[0] // 3
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:2*n*m*r].reshape(m*p, r)
        W = x[2*n*m*r:].reshape(p*n, r)
        
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Target terms
        for a, b, c, t in target_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = val - t
            obj += err * err
            grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
            grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
            grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
            
        # Zero terms
        for a, b, c, t in zero_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = val - t
            obj += err * err
            grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
            grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
            grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
            
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad

    # Try to find rank 28 decomposition
    if time.time() < deadline:
        r_target = 28
        size_U = n * m * r_target
        size_V = m * p * r_target
        size_W = p * n * r_target
        total_size = size_U + size_V + size_W
        
        for seed in range(50):
            if time.time() > deadline:
                break
            np.random.seed(seed)
            x0 = np.random.randn(total_size) * 0.1
            # Scale to help convergence?
            
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad, 
                    x0, 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success and res.fun < 1e-8:
                    x_opt = res.x
                    U = x_opt[:size_U].reshape(n*m, r_target)
                    V = x_opt[size_U:size_U+size_V].reshape(m*p, r_target)
                    W = x_opt[size_U+size_V:].reshape(p*n, r_target)
                    if validate(U, V, W, tol=1e-6):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except Exception:
                pass

    return {"U": U_sb, "V": V_sb, "W": W_sb}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
