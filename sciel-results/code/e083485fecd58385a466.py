"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen on (2,2,10) sub-block."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
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
    import random
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

    # Attempt to find a lower rank decomposition (e.g., rank 52 or 51)
    # Using random search with local optimization
    # Target rank 52
    target_rank = 52
    best_solution = None
    
    # Time budget for search
    search_start = time.time()
    max_search_time = 60.0 # seconds
    
    # Generate random initializations
    # We fix the first column of U to identity to break symmetry?
    # Or just random.
    
    for attempt in range(20):
        if time.time() - search_start > max_search_time:
            break
            
        # Random initialization
        U_init = np.random.randn(n * m, target_rank)
        V_init = np.random.randn(m * p, target_rank)
        W_init = np.random.randn(p * n, target_rank)
        
        # Define residual function
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: The tensor is defined by C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] is 1 if it's a valid product term?
        # No, the tensor M is such that C = A * B.
        # The decomposition is M = sum_r u_r v_r w_r.
        # M[i,j,k] = delta_{i,j,k} in the standard basis?
        # Actually, the tensor for matrix multiplication is:
        # T_{i j k} = 1 if we multiply A_{i j} and B_{j k} to get C_{i k}.
        # Wait, the index of W is k*n+i.
        # So W[k*n+i, r] multiplies U[i*m+j, r] and V[j*p+k, r].
        # The sum is over j.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # So the tensor entries T_{i,j,k} are 1 for all i,j,k?
        # No, the tensor is a 3rd order tensor T[i,j,k].
        # The multiplication is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
        # Here T[i,j,k] = 1 for all i,j,k.
        # So we need to decompose the tensor T where T[i,j,k]=1.
        
        def residuals(params):
            U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank + m*p*target_rank], params[n*m*target_rank + m*p*target_rank:]
            U_mat = U_flat.reshape(n*m, target_rank)
            V_mat = V_flat.reshape(m*p, target_rank)
            W_mat = W_flat.reshape(p*n, target_rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target value is 1
                        val = 0.0
                        for r in range(target_rank):
                            val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        res.append(val - 1.0)
            return res
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            sol = least_squares(residuals, params_init, method='lm', max_nfev=100)
            if sol.success and sol.cost < 1e-10:
                # Check if we can round to half-integers?
                # The task says "integer/half-integer entries".
                # If the solution is close to half-integers, we can use it.
                # Otherwise, we might need to keep floats?
                # "exactly decompose" implies the values must be exact.
                # If we use floats, they must be exact representations of half-integers.
                # So we should check if the solution is close to half-integers.
                
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
                    break # Found a valid solution, stop searching
        except Exception:
            pass
    
    if best_solution:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    return {"U": U, "V": V, "W": W}
