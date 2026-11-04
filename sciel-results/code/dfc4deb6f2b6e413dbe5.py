import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds, leaving 10 seconds margin
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try different ranks
    best_result = None
    for rank in range(30, 91):
        if time.time() > deadline:
            break
        
        # Initialize U, V, W with random values
        np.random.seed(rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Fix V, W, solve for U
            for a in range(n*m):
                # W_c' = v_c'[b] * w_c'[c] for each c'
                # t_a = T[a, :, :]
                t_a = T[a, :, :]  # (m*p, p*n)
                # G[c', c''] = sum_{b,c} V[b, c'] * W[c, c'] * V[b, c''] * W[c, c'']
                # d_a[c'] = sum_{b,c} t_a[b,c] * V[b, c'] * W[c, c']
                G = np.zeros((rank, rank))
                d = np.zeros(rank)
                for c_prime in range(rank):
                    Wc = np.outer(V[:, c_prime], W[:, c_prime])  # (m*p, p*n)
                    d[c_prime] = np.sum(t_a * Wc)
                    for c_double in range(rank):
                        Wc2 = np.outer(V[:, c_double], W[:, c_double])
                        G[c_prime, c_double] = np.sum(Wc * Wc2)
                # Solve G * u_a = d
                try:
                    u_a = np.linalg.solve(G, d)
                except np.linalg.LinAlgError:
                    u_a = np.linalg.lstsq(G, d, rcond=None)[0]
                U[a, :] = u_a
            
            # Fix U, W, solve for V
            for b in range(m*p):
                t_b = T[:, b, :]  # (n*m, p*n)
                G = np.zeros((rank, rank))
                d = np.zeros(rank)
                for c_prime in range(rank):
                    Uc = np.outer(U[:, c_prime], W[:, c_prime])  # (n*m, p*n)
                    d[c_prime] = np.sum(t_b * Uc)
                    for c_double in range(rank):
                        Uc2 = np.outer(U[:, c_double], W[:, c_double])
                        G[c_prime, c_double] = np.sum(Uc * Uc2)
                try:
                    v_b = np.linalg.solve(G, d)
                except np.linalg.LinAlgError:
                    v_b = np.linalg.lstsq(G, d, rcond=None)[0]
                V[b, :] = v_b
            
            # Fix U, V, solve for W
            for c in range(p*n):
                t_c = T[:, :, c]  # (n*m, m*p)
                G = np.zeros((rank, rank))
                d = np.zeros(rank)
                for c_prime in range(rank):
                    Uc = np.outer(U[:, c_prime], V[:, c_prime])  # (n*m, m*p)
                    d[c_prime] = np.sum(t_c * Uc)
                    for c_double in range(rank):
                        Uc2 = np.outer(U[:, c_double], V[:, c_double])
                        G[c_prime, c_double] = np.sum(Uc * Uc2)
                try:
                    w_c = np.linalg.solve(G, d)
                except np.linalg.LinAlgError:
                    w_c = np.linalg.lstsq(G, d, rcond=None)[0]
                W[:, c] = w_c
        
        # Check the error
        error = 0.0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error += (T[a, b, c] - val) ** 2
        
        if error < 1e-10:
            # Round to nearest integer/half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check the error after rounding
            error_rounded = 0.0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        val = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                        error_rounded += (T[a, b, c] - val) ** 2
            
            if error_rounded < 1e-10:
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                return best_result
    
    # If no better result found, return the schoolbook decomposition
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
