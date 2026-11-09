import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def check_solution(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def round_to_half_int(x):
        return np.round(x * 2) / 2
    
    def solve_for_W(U, V):
        R = U.shape[1]
        W = np.zeros((p*n, R))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = w
        return W
    
    def solve_for_V(U, W):
        R = U.shape[1]
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p+k, :] = v
        return V
    
    def solve_for_U(V, W):
        R = V.shape[1]
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = u
        return U
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to reduce rank using greedy approach
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(rank):
            if time.time() > deadline:
                break
            
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            U_trial = U[:, mask]
            V_trial = V[:, mask]
            W_trial = W[:, mask]
            
            W_trial = solve_for_W(U_trial, V_trial)
            V_trial = solve_for_V(U_trial, W_trial)
            U_trial = solve_for_U(V_trial, W_trial)
            
            # Round to half-integers
            U_rounded = round_to_half_int(U_trial)
            V_rounded = round_to_half_int(V_trial)
            W_rounded = round_to_half_int(W_trial)
            
            if check_solution(U_rounded, V_rounded, W_rounded):
                U, V, W = U_rounded, V_rounded, W_rounded
                rank -= 1
                improved = True
                break
    
    # If we still have a high rank, try alternating minimization with random restarts
    if rank > 30 and time.time() < deadline:
        for R in range(24, rank):
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(10):
                if time.time() > deadline:
                    break
                
                np.random.seed(restart)
                U_trial = np.random.randn(n*m, R) * 0.1
                V_trial = np.random.randn(m*p, R) * 0.1
                W_trial = np.random.randn(p*n, R) * 0.1
                
                for iteration in range(200):
                    if time.time() > deadline:
                        break
                    
                    W_trial = solve_for_W(U_trial, V_trial)
                    V_trial = solve_for_V(U_trial, W_trial)
                    U_trial = solve_for_U(V_trial, W_trial)
                    
                    U_rounded = round_to_half_int(U_trial)
                    V_rounded = round_to_half_int(V_trial)
                    W_rounded = round_to_half_int(W_trial)
                    
                    if check_solution(U_rounded, V_rounded, W_rounded):
                        U, V, W = U_rounded, V_rounded, W_rounded
                        rank = R
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
