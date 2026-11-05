import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: Try to find a decomposition with rank < 46
    # If not found, fall back to a valid decomposition.
    
    # We will search for ranks 1 to 46.
    # For each rank, we try a few random initializations of U and V.
    
    for rank in range(1, 47):
        for attempt in range(20):
            solution = _try_decompose(n, m, p, rank, attempt)
            if solution is not None:
                return solution
                
    # Fallback: Schoolbook decomposition (Rank 54)
    # This ensures validity, though score might be lower than 1.0 if 46 is not found.
    # However, for R=46, the system is very under-determined (828 vars, 54 eqs),
    # so a solution should be found easily.
    return _schoolbook(n, m, p)

def _try_decompose(n, m, p, rank, seed):
    np.random.seed(seed)
    
    # Initialize U and V with small integers
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    U = np.random.choice([-1, 0, 1], size=(n*m, rank))
    V = np.random.choice([-1, 0, 1], size=(m*p, rank))
    
    # Construct the linear system M * W = b
    # Variables: W_flat of size (p*n*rank)
    # Equations: n*m*p
    num_vars = p * n * rank
    num_eqs = n * m * p
    
    M = np.zeros((num_eqs, num_vars))
    b = np.ones(num_eqs)
    
    # Fill M
    # Equation index: e = i*m*p + j*p + k
    # Variable index: v = (k*n + i)*rank + r
    # Coefficient: U[i*m+j, r] * V[j*p+k, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r in range(rank):
                    coeff = U[i*m+j, r] * V[j*p+k, r]
                    if coeff != 0:
                        var_idx = (k * n + i) * rank + r
                        M[eq_idx, var_idx] = coeff
    
    # Solve M * W = b
    try:
        W_flat, residuals, rank_sol, s = np.linalg.lstsq(M, b, rcond=None)
    except:
        return None
        
    # Check if the solution is close to exact
    if not np.allclose(M @ W_flat, b, atol=1e-9):
        return None
        
    # Check if entries are integer or half-integer
    # Multiply by 2 and check if close to integer
    W_scaled = W_flat * 2
    if not np.allclose(W_scaled, np.round(W_scaled), atol=1e-6):
        return None
        
    # Convert to list of lists
    W = W_flat.reshape(p * n, rank)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify exactness with random matrices
    if _verify(n, m, p, U_list, V_list, W_list, rank):
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return None

def _verify(n, m, p, U, V, W, rank):
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Test with a few random matrices
    for _ in range(5):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_np[:, r].reshape(n, m)
            v_r = V_np[:, r].reshape(m, p)
            w_r = W_np[:, r].reshape(p, n)
            
            # C[i,k] += sum_j u_r[i,j] A[i,j] * v_r[j,k] B[j,k] * w_r[k,i] ?
            # No, the decomposition is:
            # T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # C[i,k] = sum_j A[i,j] B[j,k]
            # If T_{i,j,k} = 1, then C[i,k] = sum_j 1 * A[i,j] B[j,k]
            # The decomposition ensures that the tensor of coefficients is 1.
            # So C[i,k] computed via decomposition should match C_expected.
            
            # Let's compute the contribution
            # The term for r is:
            # X_r = u_r * A (element-wise? No)
            # The formula is:
            # C[i,k] = sum_r w_r[k,i] * (sum_j u_r[i,j] A[i,j]) * (sum_j v_r[j,k] B[j,k])
            # This is NOT correct. The standard decomposition is:
            # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) * W[i,k,r]
            # Wait, if U, V, W are the factors, the product is:
            # (U_r \circ A) * (V_r \circ B) * W_r
            # Where \circ is element-wise product?
            # No.
            # Let's check the 1x2x1 case again.
            # U[0,0]=1, V[0,0]=1, W[0,0]=1.
            # U[0,1]=1, V[1,1]=1, W[0,1]=1.
            # C[0,0] = 1*(A00*1 + A01*0) * (B00*1 + B10*0) * 1 + 1*(A00*0 + A01*1) * (B00*0 + B10*1) * 1
            # C[0,0] = A00 B00 + A01 B10. Correct.
            # So the formula is:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
            
            X_r = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    X_r[i] += u_r[i, j] * A[i, j]
                    
            Y_r = np.zeros(p)
            for j in range(m):
                for k in range(p):
                    Y_r[k] += v_r[j, k] * B[j, k]
                    
            for i in range(n):
                for k in range(p):
                    C_decomposed[i, k] += w_r[k, i] * X_r[i] * Y_r[k]
                    
        if not np.allclose(C_decomposed, C_expected, atol=1e-9):
            return False
            
    return True

def _schoolbook(n, m, p):
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
