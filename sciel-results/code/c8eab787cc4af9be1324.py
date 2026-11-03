import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Try ranks lower than schoolbook
    for r in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # We want to minimize the residual of the tensor equation.
            # The equation is: sum_c W[k*n+i, c] U[i'*m+j', c] V[j''*p+k', c] = delta...
            # This is equivalent to checking if the matrix multiplication tensor is decomposed.
            
            # We can define a function that computes the residual.
            # Residual is the difference between the target tensor and the decomposed tensor.
            
            # To make it efficient, we can use the fact that we only need to check a few random A, B matrices.
            # If it holds for random A, B, it likely holds for all.
            
            def get_residual(U, V, W, A, B):
                C_expected = A @ B
                C_approx = np.zeros((n, p))
                for c in range(r):
                    # u_i = sum_j A[i, j] U[i*m+j, c]
                    # v_k = sum_j B[j, k] V[j*p+k, c]
                    # C_approx[i, k] += W[k*n+i, c] * u_i * v_k
                    u = np.zeros(n)
                    v = np.zeros(p)
                    for i in range(n):
                        for j in range(m):
                            u[i] += A[i, j] * U[i * m + j, c]
                    for k in range(p):
                        for j in range(m):
                            v[k] += B[j, k] * V[j * p + k, c]
                    for i in range(n):
                        for k in range(p):
                            C_approx[i, k] += W[k * n + i, c] * u[i] * v[k]
                return np.linalg.norm(C_expected - C_approx)
            
            # Use a few random A, B to check
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            
            # Gradient descent step? Too slow to implement correctly in short time.
            # Instead, we can try to solve the linear system if we fix two matrices.
            # If we fix U and V, the equations are linear in W.
            # Let's try fixing U and V to random values and solving for W.
            
            U_fixed = np.random.randn(n * m, r)
            V_fixed = np.random.randn(m * p, r)
            
            # Construct the linear system for W.
            # For each (i, k, i', j', j'', k'), we have an equation.
            # But we can just use the random test matrices to form a linear system.
            # C_expected[i, k] = sum_c W[k*n+i, c] * (sum_j A[i, j] U[i*m+j, c]) * (sum_j B[j, k] V[j*p+k, c])
            # Let alpha_{i, c} = sum_j A[i, j] U[i*m+j, c]
            # Let beta_{k, c} = sum_j B[j, k] V[j*p+k, c]
            # Then C_expected[i, k] = sum_c W[k*n+i, c] * alpha_{i, c} * beta_{k, c}
            
            # We have n*p equations (for each i, k).
            # Unknowns are W[k*n+i, c] for c=1..r. Total n*p*r unknowns.
            # This is under-determined if n*p*r > n*p.
            # We can use multiple random A, B to get more equations.
            
            num_samples = 20
            A_samples = np.random.randn(num_samples, n, m)
            B_samples = np.random.randn(num_samples, m, p)
            
            # Build matrix M and vector b for linear system M * W_flat = b
            # W_flat is vector of length n*p*r.
            # Index of W[k*n+i, c] is (k*n+i)*r + c.
            
            M = np.zeros((num_samples * n * p, n * p * r))
            b = np.zeros(num_samples * n * p)
            
            for s in range(num_samples):
                A_s = A_samples[s]
                B_s = B_samples[s]
                C_s = A_s @ B_s
                
                alpha = np.zeros((n, r))
                beta = np.zeros((p, r))
                for c in range(r):
                    for i in range(n):
                        alpha[i, c] = sum(A_s[i, j] * U_fixed[i * m + j, c] for j in range(m))
                    for k in range(p):
                        beta[k, c] = sum(B_s[j, k] * V_fixed[j * p + k, c] for j in range(m))
                
                for i in range(n):
                    for k in range(p):
                        row_idx = s * n * p + i * p + k
                        b[row_idx] = C_s[i, k]
                        for c in range(r):
                            col_idx = (k * n + i) * r + c
                            M[row_idx, col_idx] = alpha[i, c] * beta[k, c]
            
            # Solve least squares
            try:
                W_flat, residuals, rank_m, s = np.linalg.lstsq(M, b, rcond=None)
                W_sol = W_flat.reshape(p * n, r)
                
                # Check residual
                residual = 0
                for s in range(num_samples):
                    A_s = A_samples[s]
                    B_s = B_samples[s]
                    C_s = A_s @ B_s
                    C_approx = np.zeros((n, p))
                    for c in range(r):
                        u = np.zeros(n)
                        v = np.zeros(p)
                        for i in range(n):
                            for j in range(m):
                                u[i] += A_s[i, j] * U_fixed[i * m + j, c]
                        for k in range(p):
                            for j in range(m):
                                v[k] += B_s[j, k] * V_fixed[j * p + k, c]
                        for i in range(n):
                            for k in range(p):
                                C_approx[i, k] += W_sol[k * n + i, c] * u[i] * v[k]
                    residual += np.linalg.norm(C_s - C_approx)
                
                if residual < 1e-6:
                    # Found a solution!
                    # Try to snap to rational numbers
                    # Convert to list of lists
                    U_list = [[0] * r for _ in range(n * m)]
                    V_list = [[0] * r for _ in range(m * p)]
                    W_list = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n * m):
                        for c in range(r):
                            U_list[i][c] = U_fixed[i, c]
                    for j in range(m * p):
                        for c in range(r):
                            V_list[j][c] = V_fixed[j, c]
                    for k in range(p * n):
                        for c in range(r):
                            W_list[k][c] = W_sol[k, c]
                    
                    # Snap to nearest 0.5
                    for i in range(n * m):
                        for c in range(r):
                            U_list[i][c] = round(U_list[i][c] * 2) / 2
                    for j in range(m * p):
                        for c in range(r):
                            V_list[j][c] = round(V_list[j][c] * 2) / 2
                    for k in range(p * n):
                        for c in range(r):
                            W_list[k][c] = round(W_list[k][c] * 2) / 2
                    
                    # Verify snapped solution
                    A_test2 = np.random.randn(n, m)
                    B_test2 = np.random.randn(m, p)
                    C_test2 = A_test2 @ B_test2
                    C_approx2 = np.zeros((n, p))
                    for c in range(r):
                        u = np.zeros(n)
                        v = np.zeros(p)
                        for i in range(n):
                            for j in range(m):
                                u[i] += A_test2[i, j] * U_list[i * m + j][c]
                        for k in range(p):
                            for j in range(m):
                                v[k] += B_test2[j, k] * V_list[j * p + k][c]
                        for i in range(n):
                            for k in range(p):
                                C_approx2[i, k] += W_list[k * n + i][c] * u[i] * v[k]
                    
                    if np.linalg.norm(C_test2 - C_approx2) < 1e-6:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        return best_result
            except:
                pass
                
    return best_result
