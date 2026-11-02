"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Try to find a low-rank decomposition
    for rank in range(20, 46):
        for trial in range(3):
            seed = rank * 1000 + trial * 137 + 42
            np.random.seed(seed)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            converged = False
            for iteration in range(500):
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            for r_idx in range(rank):
                                A_mat[k, r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        b_vec = np.ones(p)
                        U[i*m+j, :], res, rk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            for r_idx in range(rank):
                                A_mat[i, r_idx] = U[i*m+j, r_idx] * W[k*n+i, r_idx]
                        b_vec = np.ones(n)
                        V[j*p+k, :], res, rk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            for r_idx in range(rank):
                                A_mat[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                        b_vec = np.ones(m)
                        W[k*n+i, :], res, rk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] for r_idx in range(rank))
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    converged = True
                    break
            
            if converged:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if verify(U_list, V_list, W_list, rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
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
# EVOLVE-BLOCK-END

