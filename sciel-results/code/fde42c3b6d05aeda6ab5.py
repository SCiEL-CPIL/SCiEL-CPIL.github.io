import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    def check_solution(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check if sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] == delta[j, j']
        # We can do this by tensor contraction
        # T[i, j, j', k] = sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Contract over r
        # T[i, j, j', k] = sum_c U_reshaped[i, j, c] V_reshaped[j', k, c] W_reshaped[k, i, c]
        # This is not a simple einsum because of the indices.
        # Let's use loops for clarity and speed for small sizes
        r = U.shape[1]
        # Pre-compute products?
        # For each i, k, we need a matrix M[j, j'] = sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c]
        # This M must be the identity matrix I_m.
        
        for i in range(n):
            for k in range(p):
                # Construct M[j, j']
                # M[j, j'] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c]
                # Let v_c = V[j'*p+k, c] for all j'
                # This is a bit complex to vectorize simply without large arrays.
                # n=2, m=4, p=7. r <= 56.
                # 2*7 = 14 pairs (i, k).
                # For each pair, we compute a 4x4 matrix.
                # 14 * 4 * 4 * 56 operations = ~12k ops. Very fast.
                
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > 1e-9:
                            return False
        return True

    # Schoolbook
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
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a lower rank
    # We will try random integer matrices for ranks 50 down to 40
    # This is a heuristic search
    
    for rank in range(rank_school - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for _ in range(10):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-1, 2, size=(n*m, rank))
            V = np.random.randint(-1, 2, size=(m*p, rank))
            W = np.random.randint(-1, 2, size=(p*n, rank))
            
            if check_solution(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break # Found a solution for this rank, move to lower rank
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
