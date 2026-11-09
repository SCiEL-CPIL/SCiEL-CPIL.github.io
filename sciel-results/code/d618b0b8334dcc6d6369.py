import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    def schoolbook():
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
    
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Use broadcasting for speed
            # u_mat shape (n,m), v_mat shape (m,p), w_mat shape (p,n)
            # We want sum over r of u[i,j] v[j,k] w[k,i]
            # This is element-wise product of u[i,j], v[j,k], w[k,i]
            # u_mat is (n,m,1), v_mat is (1,m,p), w_mat is (p,1,n) -> transpose w to (1,p,n)? No.
            # w_mat is (p,n). We need w[k,i].
            # w_mat.T is (n,p). w_mat.T[i,k] is w[k,i].
            # So we need u[i,j] * v[j,k] * w_mat.T[i,k].
            # u_mat: (n,m,1)
            # v_mat: (1,m,p)
            # w_mat.T: (n,1,p)
            prod = u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat.T[:, np.newaxis, :]
            tensor += prod
        expected = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(tensor, expected, atol=1e-8)

    # Try rank 18 construction
    # Decompose into two 1x4x6 multiplications
    # Rank of 1x4x6 is 4+6-1=9
    # Total rank 18
    
    def construct_rank_18():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # For each row i in {0, 1}
        for i in range(n):
            base = i * 9
            # 1x4x6 decomposition
            # m=4, p=6
            # l=0..8
            for l in range(9):
                col = base + l
                if l < m:
                    # U_j,l = 1 if j <= l else 0
                    # V_j,k,l = 1 if j == l else 0
                    # W_k,l = 1
                    for j in range(m):
                        if j <= l:
                            U[i * m + j][col] = 1
                    for j in range(m):
                        for k in range(p):
                            if j == l:
                                V[j * p + k][col] = 1
                    for k in range(p):
                        W[k * n + i][col] = 1
                else:
                    # l >= m
                    # l' = l - m
                    # U_j,l = 1
                    # V_j,k,l = 1 if k == l' + 1 else 0
                    # W_k,l = 1 if k == l' + 1 else 0
                    lp = l - m
                    for j in range(m):
                        U[i * m + j][col] = 1
                    for j in range(m):
                        for k in range(p):
                            if k == lp + 1:
                                V[j * p + k][col] = 1
                    for k in range(p):
                        if k == lp + 1:
                            W[k * n + i][col] = 1
                            
        return {"U": U, "V": V, "W": W}

    result = construct_rank_18()
    if verify(result["U"], result["V"], result["W"]):
        return result
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
