import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def verify(U, V, W, atol=1e-9):
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(res, T, atol=atol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def try_round_and_verify(U, V, W):
        """Try rounding to integers and half-integers, verify each."""
        for scale, div in [(1, 1), (2, 2)]:
            U_r = np.round(U * scale) / div
            V_r = np.round(V * scale) / div
            W_r = np.round(W * scale) / div
            if verify(U_r, V_r, W_r, atol=1e-8):
                return U_r, V_r, W_r
        return None

    def als_round(U, V, W, max_iters=30, deadline=None):
        """Run ALS with periodic rounding checks."""
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U
            num_U = np.einsum('ijk, jr, kr -> ir', T, V, W)
            den_U = (V.T @ V) * (W.T @ W)
            try:
                U = np.linalg.solve(den_U.T, num_U.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update V
            num_V = np.einsum('ijk, ir, kr -> jr', T, U, W)
            den_V = (U.T @ U) * (W.T @ W)
            try:
                V = np.linalg.solve(den_V.T, num_V.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update W
            num_W = np.einsum('ijk, ir, jr -> kr', T, U, V)
            den_W = (U.T @ U) * (V.T @ V)
            try:
                W = np.linalg.solve(den_W.T, num_W.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Check rounding every few iterations
            if (it + 1) % 3 == 0:
                result = try_round_and_verify(U, V, W)
                if result is not None:
                    return result
        
        # Final check
        return try_round_and_verify(U, V, W)

    start_time = time.time()
    deadline = start_time + 115.0

    # Try ranks from 30 down to 25 (known bounds suggest ~27-30 for 3x3x4)
    for R in range(30, 24, -1):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Scale attempts based on remaining time
        base_attempts = 80 if R <= 27 else 40
        attempts = min(base_attempts, int(remaining / 0.5))
        attempts = max(attempts, 5)
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random values, mix of strategies
            strategy = attempt % 4
            if strategy == 0:
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
            elif strategy == 1:
                # Start closer to schoolbook structure
                U = np.random.randn(n*m, R) * 0.5 + 0.5
                V = np.random.randn(m*p, R) * 0.5 + 0.5
                W = np.random.randn(p*n, R) * 0.5 + 0.5
            elif strategy == 2:
                # Sparse-like initialization
                U = np.random.choice([0, 1, -1], size=(n*m, R)) * 0.5
                V = np.random.choice([0, 1, -1], size=(m*p, R)) * 0.5
                W = np.random.choice([0, 1, -1], size=(p*n, R)) * 0.5
            else:
                # Uniform
                U = np.random.uniform(-1, 1, (n*m, R))
                V = np.random.uniform(-1, 1, (m*p, R))
                W = np.random.uniform(-1, 1, (p*n, R))
            
            # Run ALS with increased iterations
            result = als_round(U, V, W, max_iters=25, deadline=deadline)
            if result is not None:
                U_r, V_r, W_r = result
                # Final high-precision polish
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    num_U = np.einsum('ijk, jr, kr -> ir', T, V_r, W_r)
                    den_U = (V_r.T @ V_r) * (W_r.T @ W_r)
                    try:
                        U_r = np.linalg.solve(den_U.T, num_U.T).T
                    except np.linalg.LinAlgError:
                        pass
                    num_V = np.einsum('ijk, ir, kr -> jr', T, U_r, W_r)
                    den_V = (U_r.T @ U_r) * (W_r.T @ W_r)
                    try:
                        V_r = np.linalg.solve(den_V.T, num_V.T).T
                    except np.linalg.LinAlgError:
                        pass
                    num_W = np.einsum('ijk, ir, jr -> kr', T, U_r, V_r)
                    den_W = (U_r.T @ U_r) * (V_r.T @ V_r)
                    try:
                        W_r = np.linalg.solve(den_W.T, num_W.T).T
                    except np.linalg.LinAlgError:
                        pass
                    result2 = try_round_and_verify(U_r, V_r, W_r)
                    if result2 is not None:
                        U_r, V_r, W_r = result2
                        break
                if verify(U_r, V_r, W_r, atol=1e-10):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }

    return get_schoolbook()
