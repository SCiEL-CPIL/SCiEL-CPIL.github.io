import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n=2, m=3, p=6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The multiplication tensor T[i,j,k] = 1 if i is row of A, j is col of A/row of B, k is col of B
    # Actually, the tensor is: T[a_row, b_col, a_col_b_row] where the multiplication is:
    # C[a_row, b_col] = sum_{a_col} A[a_row, a_col] * B[a_col, b_col]
    # So T[a_row, b_col, a_col] = 1
    
    # In the factorization: T[i,j,k] = sum_r U[i,r] * V[k,r] * W[j,r]
    # where i indexes A (n*m), k indexes B (m*p), j indexes C (p*n)
    
    rank = U.shape[1]
    T_reconstructed = np.einsum('ir,kr,jr->ijk', U, V, W)
    
    # Build the true tensor
    T_true = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_col in range(p):
                # A index: a_row * m + a_col
                # B index: a_col * p + b_col  
                # C index: b_col * n + a_row
                T_true[a_row * m + a_col, a_col * p + b_col, b_col * n + a_row] = 1
    
    diff = np.max(np.abs(T_reconstructed - T_true))
    return diff

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # First, try the known rank-33 solution
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
        U[3][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
    
    for x in range(3):
        c0 = 2*x; c1 = 2*x+1
        v_b1_c0 = 6+c0; v_b1_c1 = 6+c1
        v_b2_c0 = 12+c0; v_b2_c1 = 12+c1
        w_c0_0 = 2*c0; w_c0_1 = 2*c0+1
        w_c1_0 = 2*c1; w_c1_1 = 2*c1+1
        
        U[1][col]=1; U[5][col]=1; V[v_b1_c0][col]=1; V[v_b2_c1][col]=1; W[w_c0_0][col]=1; W[w_c1_1][col]=1; col+=1
        U[4][col]=1; U[5][col]=1; V[v_b1_c0][col]=1; W[w_c0_1][col]=1; W[w_c1_1][col]=-1; col+=1
        U[1][col]=1; V[v_b1_c1][col]=1; V[v_b2_c1][col]=-1; W[w_c1_0][col]=1; W[w_c1_1][col]=1; col+=1
        U[5][col]=1; V[v_b2_c0][col]=1; V[v_b1_c0][col]=-1; W[w_c0_0][col]=1; W[w_c0_1][col]=1; col+=1
        U[1][col]=1; U[2][col]=1; V[v_b2_c1][col]=1; W[w_c0_0][col]=-1; W[w_c1_0][col]=1; col+=1
        U[4][col]=1; U[1][col]=-1; V[v_b1_c0][col]=1; V[v_b1_c1][col]=1; W[w_c1_1][col]=1; col+=1
        U[2][col]=1; U[5][col]=-1; V[v_b2_c0][col]=1; V[v_b2_c1][col]=1; W[w_c0_0][col]=1; col+=1
    
    # Verify
    err = verify_decomposition(U, V, W, n, m, p)
    if err > 1e-9:
        # Fallback to trivial
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for a_row in range(n):
            for a_col in range(m):
                for b_col in range(p):
                    U[a_row*m+a_col][col] = 1
                    V[a_col*p+b_col][col] = 1
                    W[b_col*n+a_row][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank via numerical search
    # The tensor T has shape (n*m, m*p, p*n) = (6, 18, 12)
    # We want to find r such that T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s
    
    # Try rank 32, 31, ...
    for target_rank in range(32, 25, -1):
        if time.time() > deadline - 30:
            break
        
        # Use random initialization and gradient descent
        best_err = np.inf
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(attempt * 42 + target_rank)
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Build target tensor
            T_true = np.zeros((n*m, m*p, p*n))
            for a_row in range(n):
                for a_col in range(m):
                    for b_col in range(p):
                        T_true[a_row*m+a_col, a_col*p+b_col, b_col*n+a_row] = 1
            
            def loss(params):
                U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                T_rec = np.einsum('ir,kr,jr->ijk', U_p, V_p, W_p)
                return np.sum((T_rec - T_true)**2)
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(loss, params0, method='L-BFGS-B', 
                                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < best_err:
                    best_err = result.fun
                    if best_err < 1e-10:
                        # Found a solution!
                        U_p = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                        V_p = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                        W_p = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                        
                        # Convert to list format
                        U_list = [[float(U_p[i,j]) for j in range(target_rank)] for i in range(n*m)]
                        V_list = [[float(V_p[i,j]) for j in range(target_rank)] for i in range(m*p)]
                        W_list = [[float(W_p[i,j]) for j in range(target_rank)] for i in range(p*n)]
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
    
    return {"U": U, "V": V, "W": W}
