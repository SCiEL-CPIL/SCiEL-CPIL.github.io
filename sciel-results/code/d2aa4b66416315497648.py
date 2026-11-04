import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Target tensor equations:
    # For all i in [n], j in [m], k in [p]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Precompute index mappings for efficiency
    # U indices: i, j -> i*m+j
    # V indices: j, k -> j*p+k
    # W indices: k, i -> k*n+i
    
    # Try to find a decomposition with rank r
    # Start from schoolbook rank n*m*p and go down
    max_rank = n * m * p
    
    # Helper to check residual
    def calc_residual(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Residual for each (i, j, k)
        res = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    if val != 1.0:
                        res += (val - 1.0) ** 2
        return np.sqrt(res)

    # Helper to run ALS
    def run_als(r, init_type='random'):
        # Initialize U, V, W
        if init_type == 'schoolbook':
            # Start with schoolbook and remove some columns?
            # Easier to just use random for now, or truncated schoolbook
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            # Fill first r columns of schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r: break
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
                    if col >= r: break
                if col >= r: break
        else:
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
        
        for _ in range(100):
            if time.time() > deadline:
                return None
            
            # Update U
            # For each (i, j), solve A * u = b
            # Equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
            # Coeff for U[i,j,r] in eq (i,j,k) is V[j,k,r] * W[k,i,r]
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    b = np.ones(p)
                    for k in range(p):
                        # Row k of A corresponds to equation for (i, j, k)
                        # Col r of A corresponds to variable U[i, j, r]
                        # Coeff is V[j, k, r] * W[k, i, r]
                        v_row = V[j * p + k]
                        w_row = W[k * n + i]
                        A[k] = v_row * w_row
                    # Solve A * U[i, j, :] = b
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = sol
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    b = np.ones(n)
                    for i in range(n):
                        u_row = U[i * m + j]
                        w_row = W[k * n + i]
                        A[i] = u_row * w_row
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = sol
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        A[j] = u_row * v_row
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = sol
            
            # Check residual
            if calc_residual(U, V, W) < 1e-8:
                return U, V, W
                
        return None

    # Search for minimum rank
    best_result = None
    for r in range(max_rank - 1, 90, -1):
        if time.time() > deadline:
            break
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            res = run_als(r, init_type='random')
            if res is not None:
                U, V, W = res
                # Convert to list of lists for return
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not found with random, try schoolbook init?
        # Schoolbook init might be stuck in local minima if r < 120
        # But let's try one restart with schoolbook init
        if time.time() < deadline:
            res = run_als(r, init_type='schoolbook')
            if res is not None:
                U, V, W = res
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    rank = max_rank
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
# EVOLVE-BLOCK-END
