"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen on (2,2,10) sub-block."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build the known rank-55 decomposition
    # Split j into {0,1} (Strassen per k-pair) and j=2 (schoolbook)
    # (2,2,10): 5 k-pairs × 7 Strassen = 35
    # (2,1,10): 2×10 = 20
    # Total: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    
    # Strassen for (2,2,2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0][col] = 1   # A[0,0], row 0*3+0
        U[4][col] = 1   # A[1,1], row 1*3+1
        V[k0][col] = 1       # B[0,k0], row 0*10+k0
        V[10+k1][col] = 1    # B[1,k1], row 1*10+k1
        W[2*k0][col] = 1     # C[0,k0], row k0*2+0
        W[2*k1+1][col] = 1   # C[1,k1], row k1*2+1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3][col] = 1   # A[1,0], row 1*3+0
        U[4][col] = 1   # A[1,1], row 1*3+1
        V[k0][col] = 1  # B[0,k0], row 0*10+k0
        W[2*k0+1][col] = 1    # C[1,k0], row k0*2+1
        W[2*k1+1][col] = -1   # C[1,k1], row k1*2+1
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0][col] = 1   # A[0,0], row 0*3+0
        V[k1][col] = 1       # B[0,k1], row 0*10+k1
        V[10+k1][col] = -1   # B[1,k1], row 1*10+k1
        W[2*k1][col] = 1     # C[0,k1], row k1*2+0
        W[2*k1+1][col] = 1   # C[1,k1], row k1*2+1
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4][col] = 1   # A[1,1], row 1*3+1
        V[10+k0][col] = 1   # B[1,k0], row 1*10+k0
        V[k0][col] = -1     # B[0,k0], row 0*10+k0
        W[2*k0][col] = 1    # C[0,k0], row k0*2+0
        W[2*k0+1][col] = 1  # C[1,k0], row k0*2+1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] = 1   # A[0,0], row 0*3+0
        U[1][col] = 1   # A[0,1], row 0*3+1
        V[10+k1][col] = 1  # B[1,k1], row 1*10+k1
        W[2*k0][col] = -1  # C[0,k0], row k0*2+0
        W[2*k1][col] = 1   # C[0,k1], row k1*2+0
        col += 1
        
        # M6 = (A[1,1]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[4][col] = 1   # A[1,1], row 1*3+1
        U[0][col] = -1  # A[0,0], row 0*3+0
        V[k0][col] = 1     # B[0,k0], row 0*10+k0
        V[k1][col] = 1     # B[0,k1], row 0*10+k1
        W[2*k1+1][col] = 1  # C[1,k1], row k1*2+1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] = 1   # A[0,1], row 0*3+1
        U[4][col] = -1  # A[1,1], row 1*3+1
        V[10+k0][col] = 1  # B[1,k0], row 1*10+k0
        V[10+k1][col] = 1  # B[1,k1], row 1*10+k1
        W[2*k0][col] = 1    # C[0,k0], row k0*2+0
        W[2*k1+1][col] = -1 # C[1,k1], row k1*2+1
        col += 1
    
    # Part 2: Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1   # A[i,2], row i*3+2
            V[20+k][col] = 1    # B[2,k], row 2*10+k
            W[k*2+i][col] = 1   # C[i,k], row k*2+i
            col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verify with multiple random tests
    random.seed(42)
    def verify(U, V, W, n, m, p, max_checks=20):
        rank = len(U[0])
        for _ in range(max_checks):
            A = [[random.uniform(-2, 2) for _ in range(m)] for _ in range(n)]
            B = [[random.uniform(-2, 2) for _ in range(p)] for _ in range(m)]
            C = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C[i][k] += A[i][j] * B[j][k]
            Cc = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0.0
                    for r in range(rank):
                        ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                        vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                        acc += W[k*n+i][r] * ua * vb
                    Cc[i][k] = acc
            for i in range(n):
                for k in range(p):
                    if abs(C[i][k] - Cc[i][k]) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    # Attempt to find a lower rank decomposition
    # Try rank 54 and 53 with a structured approach
    # Key idea: try to merge some of the schoolbook terms with Strassen terms
    # or find a more efficient decomposition
    
    # First, let's try a direct optimization approach for rank 54
    # We'll use a smart initialization based on the rank-55 solution
    # and try to find a valid rank-54 decomposition
    
    best_solution = None
    
    # Time budget for search
    search_start = time.time()
    max_search_time = 80.0  # seconds
    
    # Try rank 54 first
    for target_rank in [54, 53]:
        if time.time() - search_start > max_search_time:
            break
        
        # Smart initialization: start from rank-55 and try to merge columns
        # Or use random initialization with scaling
        
        for attempt in range(15):
            if time.time() - search_start > max_search_time:
                break
            
            # Random initialization with moderate scaling
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            # Define residual function - vectorized for speed
            def residuals(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank:]
                
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                
                # Compute all tensor entries
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Target: T[i,j,k] = 1 for all i,j,k
                res = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res[i, j, k] = np.sum(
                                U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :]
                            )
                return (res - 1.0).flatten()
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residuals, params_init, method='lm', max_nfev=30)
                if sol.success and sol.cost < 1e-10:
                    U_sol = sol.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = sol.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = sol.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify rounded solution
                    if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                        best_solution = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                        break
            except Exception:
                pass
        
        if best_solution:
            break
    
    if best_solution:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # If no better solution found, return the rank-55 solution
    return {"U": U, "V": V, "W": W}
