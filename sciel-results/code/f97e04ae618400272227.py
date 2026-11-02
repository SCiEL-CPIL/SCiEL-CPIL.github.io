"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    dim_U = n * m  # 9
    dim_V = m * p  # 12
    dim_W = p * n  # 12
    
    # Build target tensor
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        T_rec = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
        return np.allclose(T_rec, T, atol=tol, rtol=0)
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat.tolist()]
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition via systematic search
    def try_systematic(r, n_trials=400):
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            V_mat = np.zeros((dim_V, r))
            W_mat = np.zeros((dim_W, r))
            
            for c in range(r):
                n_nz = np.random.randint(2, 5)
                indices = np.random.choice(dim_V, size=n_nz, replace=False)
                for idx in indices:
                    V_mat[idx, c] = np.random.choice([-1, 1])
                
                n_nz = np.random.randint(2, 5)
                indices = np.random.choice(dim_W, size=n_nz, replace=False)
                for idx in indices:
                    W_mat[idx, c] = np.random.choice([-1, 1])
            
            U_mat = np.zeros((dim_U, r))
            valid = True
            for a in range(dim_U):
                M = np.zeros((dim_V * dim_W, r))
                rhs = np.zeros(dim_V * dim_W)
                for b in range(dim_V):
                    for cc in range(dim_W):
                        M[b * dim_W + cc, :] = V_mat[b, :] * W_mat[cc, :]
                        rhs[b * dim_W + cc] = T[a, b, cc]
                
                try:
                    sol, residuals, rank_M, sv = np.linalg.lstsq(M, rhs, rcond=1e-10)
                except:
                    valid = False
                    break
                
                if not np.allclose(M @ sol, rhs, atol=1e-8):
                    valid = False
                    break
                
                sol_rounded = np.round(sol * 2) / 2
                if not np.allclose(sol_rounded, sol, atol=1e-6):
                    valid = False
                    break
                
                U_mat[a, :] = sol_rounded
            
            if valid:
                if verify(U_mat, V_mat, W_mat, tol=1e-9):
                    return (U_mat, V_mat, W_mat)
        
        return None
    
    for r in range(30, 36):
        if time.time() > deadline:
            break
        result = try_systematic(r, n_trials=300)
        if result is not None:
            U_mat, V_mat, W_mat = result
            return {"U": to_list(U_mat), "V": to_list(V_mat), "W": to_list(W_mat)}
    
    return schoolbook()
# EVOLVE-BLOCK-END
