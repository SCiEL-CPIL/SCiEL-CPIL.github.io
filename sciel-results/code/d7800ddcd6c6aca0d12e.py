import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook fallback
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def is_valid_half_integer_solution(U, V, W):
        # Check if all entries are integers or half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not (abs(val * 2 - round(val * 2)) < 1e-9):
                        return False
        # Verify tensor identity
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    start_time = time.time()
    deadline = 140
    
    for rank in range(27, rank_school):
        if time.time() - start_time > deadline:
            break
        
        found = False
        for attempt in range(30):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        b = np.ones(p)
                        u_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = u_row
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            for c in range(rank):
                                A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        b = np.ones(n)
                        v_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = v_row
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        b = np.ones(m)
                        w_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = w_row
                
                # Check convergence
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            residual += (val - 1.0) ** 2
                
                if residual < 1e-10:
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    
                    if is_valid_half_integer_solution(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
                    break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
