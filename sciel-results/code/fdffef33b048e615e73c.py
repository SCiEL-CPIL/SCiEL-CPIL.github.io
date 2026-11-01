import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 56
    
    def check_solution(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > 1e-9:
                            return False
        return True
    
    # Schoolbook solution
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Use a more structured search: 
    # The tensor T[i,j,jp,k] = delta[j,jp]
    # This means for each (i,k), the (j,jp) slice is I_m.
    
    # Let's try to find a decomposition using the structure.
    # We can write the tensor as a sum of rank-1 tensors.
    # A rank-1 tensor has the form u_i * v_j * w_jp * x_k (but with the specific index structure).
    
    # Actually, let's use a different representation.
    # The bilinear map is: C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_c U_c[i,j] A[i,j] V_c[j,k] B[j,k] W_c[i,k]
    # Wait, the convention is:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] * A[i,j] * B[j',k] = C[i,k]
    # And this should equal sum_j A[i,j] B[j,k] (with j' = j)
    
    # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta[j,j] for the correct j,
    # and sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = 0 for j != j'.
    
    # This is a system of equations. Let's set it up as a linear system.
    
    # Variables: U[i*m+j, c], V[j'*p+k, c], W[k*n+i, c] for c=0..r-1
    # But these are multiplied together, so it's nonlinear.
    
    # Alternative: Fix two of the matrices and solve for the third.
    # If we fix U and V, then for each (i,k,j,jp):
    # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[jp*p+k, c]) = delta[j,jp]
    # This is linear in W!
    
    # So: For fixed U, V, we can solve for W via linear least squares.
    # Similarly, fix U, W and solve for V, or fix V, W and solve for U.
    
    # Let's use this: Start with random U, V, solve for W, check if solution exists.
    
    np.random.seed(123)
    
    # Build the linear system for W given U, V
    # For each (i, k, j, jp):
    # sum_c W[k*n+i, c] * M_c[i,k,j,jp] = delta[j,jp]
    # where M_c[i,k,j,jp] = U[i*m+j, c] * V[jp*p+k, c]
    
    # This is: A_W * W = b_W
    # where A_W has rows indexed by (i,k,j,jp) and columns by (k*n+i, c)
    # But W is indexed by (k*n+i, c), so for each (k*n+i), we have r variables.
    
    # Let's group by (k*n+i):
    # For fixed (i,k), we have m*m equations (for j,jp) and r unknowns (W[k*n+i, c]).
    # The equation is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c] = delta[j,jp]
    
    # Let's define for fixed (i,k):
    # Let u_jc = U[i*m+j, c] for j=0..m-1, c=0..r-1
    # Let v_jpc = V[jp*p+k, c] for jp=0..m-1, c=0..r-1
    # Then: sum_c W_c * u_jc * v_jpc = delta[j,jp]
    
    # This is: (u * v^T) * W = delta_vec
    # where u is (m x r), v is (m x r), W is (r,)
    # u * v^T is (m x m) matrix with entries (u*v^T)[j,jp] = sum_c u[j,c] * v[jp,c]
    # So: (u*v^T) * W = e_j... wait, this doesn't quite work.
    
    # Let me re-examine:
    # sum_c W_c * U[i*m+j, c] * V[jp*p+k, c] = delta[j,jp]
    # For fixed i, k, this is a system in W_c:
    # sum_c W_c * (U[i*m+j, c] * V[jp*p+k, c]) = delta[j,jp] for all j,jp
    
    # Let A_{(j,jp), c} = U[i*m+j, c] * V[jp*p+k, c]
    # Then A * W = b where b_{j,jp} = delta[j,jp]
    # A is (m*m x r), W is (r,), b is (m*m,)
    
    # For this to have a solution, we need rank(A) = rank([A|b]) and b in range(A).
    
    # Let's try: random U, V, then for each (i,k), check if the system is consistent.
    
    for rank in range(55, 44, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
            
            W = np.zeros((p*n, rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A matrix: (m*m x rank)
                    A = np.zeros((m*m, rank))
                    for j in range(m):
                        for jp in range(m):
                            A[j*m+jp, :] = U[i*m+j, :] * V[jp*p+k, :]
                    
                    b = np.zeros(m*m)
                    for j in range(m):
                        b[j*m+j] = 1.0
                    
                    # Solve A * W_vec = b
                    # Check if consistent
                    try:
                        W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if A * W_vec ≈ b
                        if np.max(np.abs(A @ W_vec - b)) > 1e-7:
                            valid = False
                            break
                        W[k*n+i, :] = W_vec
                    except:
                        valid = False
                        break
            
            if valid:
                # Verify the full solution
                if check_solution(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = rank
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
