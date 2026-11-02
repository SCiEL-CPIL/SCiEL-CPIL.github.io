import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    best_result = None
    best_rank = n * m * p # 72
    
    # Fallback: Schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    start_time = time.time()
    deadline = start_time + 110 # 110s limit

    def solve_for_U(V, W, R):
        U_sol = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                b = np.ones(p)
                for k in range(p):
                    for r in range(R):
                        A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_sol[i*m+j, :] = sol
                except:
                    U_sol[i*m+j, :] = 0
        return U_sol

    def solve_for_V(U, W, R):
        V_sol = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    for r in range(R):
                        A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_sol[j*p+k, :] = sol
                except:
                    V_sol[j*p+k, :] = 0
        return V_sol

    def solve_for_W(U, V, R):
        W_sol = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    for r in range(R):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_sol[k*n+i, :] = sol
                except:
                    W_sol[k*n+i, :] = 0
        return W_sol

    def check_residual(U, V, W, R):
        res = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res += (val - 1)**2
        return res

    found_solution = False
    # Try ranks from 20 to 40
    for R in range(20, 41):
        if found_solution or time.time() > deadline:
            break
        
        for restart in range(30):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-2, 3, (n*m, R)) * 0.5
            V = np.random.randint(-2, 3, (m*p, R)) * 0.5
            W = np.random.randint(-2, 3, (p*n, R)) * 0.5
            
            for it in range(15):
                if time.time() > deadline:
                    break
                U = solve_for_U(V, W, R)
                V = solve_for_V(U, W, R)
                W = solve_for_W(U, V, R)
                
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                res = check_residual(U_rounded, V_rounded, W_rounded, R)
                if res < 1e-4:
                    if R < best_rank:
                        best_rank = R
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    found_solution = True
                    break

    return best_result
