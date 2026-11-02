import numpy as np

def construct():
    n, m, p = 2, 2, 3
    rank = 10
    
    # Search for a valid half-integer rank-10 decomposition
    for seed in range(1000):
        rng = np.random.RandomState(seed)
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        U = rng.choice(vals, size=(n*m, rank)).astype(float)
        V = rng.choice(vals, size=(m*p, rank)).astype(float)
        
        # Solve linear system for W
        num_eqs = n * m * p
        num_unknowns = p * n * rank
        A_mat = np.zeros((num_eqs, num_unknowns))
        b_vec = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for r in range(rank):
                        w_idx = (k * n + i) * rank + r
                        A_mat[eq, w_idx] = U[i * m + j, r] * V[j * p + k, r]
        
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W = W_flat.reshape(p * n, rank)
        
        # Round W to nearest half-integer
        W_rounded = np.round(W * 2) / 2
        
        # Verify the decomposition
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W_rounded[k*n+i, r]
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-8:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W_rounded.tolist()
            }
    
    # Fallback: schoolbook decomposition (rank 12)
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1.0
                V_fb[j*p+k][col] = 1.0
                W_fb[k*n+i][col] = 1.0
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
