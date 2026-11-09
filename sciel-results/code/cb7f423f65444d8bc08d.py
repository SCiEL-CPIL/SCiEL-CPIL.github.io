import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrix multiplication using 3x3 grid of 2x2 Strassen blocks
    # Total rank: 27 block multiplications * 7 Strassen terms = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Iterate over 3x3 block grid
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for u_coeffs, v_coeffs, w_coeffs in terms:
                    for local_idx, coeff in u_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * k + c
                        U[global_r * m + global_c][col] += coeff
                    
                    for local_idx, coeff in v_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * k + r
                        global_c = 2 * j + c
                        V[global_r * p + global_c][col] += coeff
                    
                    for local_idx, coeff in w_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * j + c
                        W[global_c * n + global_r][col] += coeff
                    
                    col += 1
    
    # Verify correctness with a random test (quick check)
    # Convert to numpy arrays for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape: U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    # The tensor decomposition says:
    # T[i,k,j] = sum_t U[i,k,t] * V[k,j,t] * W[j,i,t]
    # should equal delta_{ik} delta_{kj} for matrix mult tensor... 
    # Actually for matrix multiplication: C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor T[i,k,j] = 1 if this is a valid (i,k,j) triple for the product
    # i.e., T[i,k,j] = 1 for all i,k,j (since every (i,k,j) contributes)
    # Wait, the matrix multiplication tensor has T[i,k,j] = 1 for all i,k,j in [0..n-1]x[0..m-1]x[0..p-1]
    # The decomposition: sum_t U[i,k,t] V[k,j,t] W[j,i,t] = 1 for all i,k,j
    
    # Let's verify: compute the tensor and check it's all 1s
    # T[i,k,j] = sum_t U[i,k,t] * V[k,j,t] * W[j,i,t]
    
    # Efficient computation using einsum
    # U_t: (n, m, rank), V_t: (m, p, rank), W_t: (p, n, rank)
    # T[i,k,j] = sum_t U_t[i,k,t] * V_t[k,j,t] * W_t[j,i,t]
    
    # This is a bit tricky to compute directly. Let's do it in steps.
    # First: M[i,k,j] = sum_t U_t[i,k,t] * V_t[k,j,t] -> shape (n, m, p)
    M = np.einsum('ikt,kjt->ijk', U_t, V_t)
    # Then: T[i,k,j] = sum_t ... wait, we already summed over t.
    # Actually T[i,k,j] = sum_t U[i,k,t] * V[k,j,t] * W[j,i,t]
    # = sum_t (U[i,k,t] * V[k,j,t]) * W[j,i,t]
    # Let's compute it properly:
    
    # T[i,k,j] = sum_t U_t[i,k,t] * V_t[k,j,t] * W_t[j,i,t]
    # We can compute: for each t, add U_t[:,:,t] * V_t[:,:,t] * W_t[:,:,t]
    # But that's rank many terms. Let's use einsum differently.
    
    # T[i,k,j] = sum_t U[i,k,t] * V[k,j,t] * W[j,i,t]
    # = einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t)
    
    T = np.einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t)
    
    # For matrix multiplication, T[i,k,j] should be 1 for all i,k,j
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-10):
        # Fallback to schoolbook if Strassen verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1
                    V_s[j * p + k][col_s] = 1
                    W_s[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
