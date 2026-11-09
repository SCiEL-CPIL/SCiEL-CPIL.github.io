import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        # Check: for all a, b, c, d:
        # If b != c: sum_alpha U[a*m+b, alpha] V[c*p+d, alpha] W[d*n+i, alpha] = 0 for all i
        # If b == c: sum_alpha U[a*m+b, alpha] V[b*p+k, alpha] W[k*n+i, alpha] = delta_{i, a} for all i, k
        
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    if b == c:
                        # For each k, the column should be e_a
                        for k in range(p):
                            for i in range(n):
                                val = np.sum(U[a * m + b, :] * V[b * p + k, :] * W[k * n + i, :])
                                expected = 1.0 if i == a else 0.0
                                if abs(val - expected) > 1e-8:
                                    return False
                    else:
                        # For each d, the sum should be 0 for all i
                        for d in range(p):
                            for i in range(n):
                                val = np.sum(U[a * m + b, :] * V[c * p + d, :] * W[d * n + i, :])
                                if abs(val) > 1e-8:
                                    return False
        return True

    def residual_vec(x, rank):
        """Compute the residual for a given parameter vector."""
        U = x[:n * m * rank].reshape(n * m, rank)
        V = x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
        W = x[n * m * rank + m * p * rank:].reshape(p * n, rank)
        
        res = np.zeros(n * m * (m - 1) * p * n + n * m * p * n)
        idx = 0
        
        # b != c constraints: must be 0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    if b == c:
                        continue
                    for d in range(p):
                        for i in range(n):
                            res[idx] = np.sum(U[a * m + b, :] * V[c * p + d, :] * W[d * n + i, :])
                            idx += 1
        
        # b == c constraints: must be delta_{i, a}
        for a in range(n):
            for b in range(m):
                for k in range(p):
                    for i in range(n):
                        val = np.sum(U[a * m + b, :] * V[b * p + k, :] * W[k * n + i, :])
                        expected = 1.0 if i == a else 0.0
                        res[idx] = val - expected
                        idx += 1
        
        return res

    # Try to find lower rank decompositions
    for rank in range(80, 29, -1):
        if time.time() > deadline - 15:
            break
            
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            # Initialize with schoolbook pattern truncated to this rank
            U0 = np.zeros((n * m, rank))
            V0 = np.zeros((m * p, rank))
            W0 = np.zeros((p * n, rank))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= rank:
                            break
                        U0[i * m + j, col] = 1
                        V0[j * p + k, col] = 1
                        W0[k * n + i, col] = 1
                        col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            if col >= rank:
                break
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Add small random perturbation
            x0 += np.random.randn(len(x0)) * 0.01
            
            try:
                from scipy.optimize import least_squares
                res = least_squares(residual_vec, x0, args=(rank,), method='lm', max_nfev=200)
                if res.cost < 1e-10:
                    U = res.x[:n * m * rank].reshape(n * m, rank)
                    V = res.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                    W = res.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                    
                    # Verify the solution
                    if verify(U, V, W):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        break  # Try next rank
            except Exception:
                pass
    
    return best_result
