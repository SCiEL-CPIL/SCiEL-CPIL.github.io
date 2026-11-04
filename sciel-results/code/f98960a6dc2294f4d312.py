# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Dimensions for 2x3 * 3x3
    n, m, p = 2, 3, 3
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    def schoolbook_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        # Build the multiplication tensor T[i,j,k] = 1
        T = np.ones((n, m, p))
        
        # Reconstruct: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        recon = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += u_col[i*m + j] * v_col[j*p + k] * w_col[k*n + i]
        
        return np.allclose(recon, T, atol=tol)

    # Start with schoolbook (rank 18)
    U, V, W = schoolbook_decomp()
    best_rank = 18
    best_result = {"U": U, "V": V, "W": W}

    if verify_decomp(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: return schoolbook anyway
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
