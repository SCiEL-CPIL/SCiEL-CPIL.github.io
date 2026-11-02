import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 3, 6
    
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
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def solve_for_W(U, V, r):
        # U is (nm x r), V is (mp x r)
        # We want W (pn x r)
        # Equation: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
        # For each (i, k), we have a system of m equations for W[k*n+i, :]
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                # Construct the coefficient matrix for this (i,k)
                # Rows correspond to j in 0..m-1
                # Cols correspond to c in 0..r-1
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                b = np.ones(m)
                # Solve A @ w = b with minimum norm solution
                try:
                    # Use pinv for better conditioning
                    w = np.linalg.pinv(A) @ b
                except:
                    w = np.zeros(r)
                W[k * n + i, :] = w
        return W

    def solve_for_U(V, W, r):
        # V is (mp x r), W is (pn x r)
        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For each (i, j), we have a system of p equations for U[i*m+j, :]
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                b = np.ones(p)
                try:
                    u = np.linalg.pinv(A) @ b
                except:
                    u = np.zeros(r)
                U[i * m + j, :] = u
        return U

    def solve_for_V(U, W, r):
        # U is (nm x r), W is (pn x r)
        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For each (j, k), we have a system of n equations for V[j*p+k, :]
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                b = np.ones(n)
                try:
                    v = np.linalg.pinv(A) @ b
                except:
                    v = np.zeros(r)
                V[j * p + k, :] = v
        return V

    def compute_error(U, V, W):
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0) ** 2
        return err

    # Search for lower rank
    for rank in range(18, best_rank):
        if time.time() > deadline - 15:
            break
            
        found = False
        # Try multiple random initializations with different strategies
        for init in range(80):
            if time.time() > deadline - 8:
                break
                
            # Different initialization strategies
            if init % 3 == 0:
                # Standard random
                U = np.random.randn(n * m, rank)
                V = np.random.randn(m * p, rank)
                W = np.random.randn(p * n, rank)
            elif init % 3 == 1:
                # Sparse initialization
                U = np.random.randn(n * m, rank)
                U[np.random.rand(n * m, rank) < 0.5] = 0
                V = np.random.randn(m * p, rank)
                V[np.random.rand(m * p, rank) < 0.5] = 0
                W = np.random.randn(p * n, rank)
                W[np.random.rand(p * n, rank) < 0.5] = 0
            else:
                # Binary initialization
                U = np.random.randint(0, 2, (n * m, rank)) * 2 - 1
                V = np.random.randint(0, 2, (m * p, rank)) * 2 - 1
                W = np.random.randint(0, 2, (p * n, rank)) * 2 - 1
            
            # Iterate with adaptive stopping
            prev_err = float('inf')
            for it in range(150):
                W = solve_for_W(U, V, rank)
                U = solve_for_U(V, W, rank)
                V = solve_for_V(U, W, rank)
                
                err = compute_error(U, V, W)
                
                # Check for convergence
                if err < 1e-10:
                    if verify(U, V, W):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        found = True
                        break
                
                # Early stop if no progress
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
            if found:
                break
        if found:
            break
            
    return best_result
