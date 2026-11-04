import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # DFT matrix for the middle dimension m=3
    # F[omega, j] = exp(2*pi*i*omega*j/m)
    omegas = np.arange(m)
    js = np.arange(m)
    F = np.exp(2j * np.pi * np.outer(omegas, js) / m)
    
    # We use the decomposition:
    # C = (1/m) * sum_omega (A * F[omega, :].conj().T) @ (F[omega, :] * B.T).T
    # Actually, let's stick to the scalar form derived:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # = sum_omega sum_j sum_j' A[i, j] B[j', k] (1/m) F^*[omega, j] F[omega, j']
    # = sum_omega (sum_j A[i, j] (1/sqrt(m)) F^*[omega, j]) * (sum_j' B[j', k] (1/sqrt(m)) F[omega, j'])
    
    # We construct rank-1 tensors for each omega, i, k
    # U[i, j, r] = delta[i, i_r] * (1/sqrt(m)) * F^*[omega_r, j]
    # V[j, k, r] = delta[k, k_r] * (1/sqrt(m)) * F[omega_r, j]
    # W[k, i, r] = delta[k, k_r] * delta[i, i_r] * sqrt(m)
    
    # Rank = m * n * p
    rank = m * n * p
    
    U = np.zeros((n * m, rank), dtype=complex)
    V = np.zeros((m * p, rank), dtype=complex)
    W = np.zeros((p * n, rank), dtype=complex)
    
    col = 0
    for omega in range(m):
        for i in range(n):
            for k in range(p):
                # U part: depends on i, j
                for j in range(m):
                    U[i * m + j, col] = np.conj(F[omega, j]) / np.sqrt(m)
                
                # V part: depends on j, k
                for j in range(m):
                    V[j * p + k, col] = F[omega, j] / np.sqrt(m)
                
                # W part: depends on k, i
                W[k * n + i, col] = np.sqrt(m)
                
                col += 1
    
    # The result is complex due to DFT.
    # The problem requires integer/half-integer entries.
    # We can convert to real by using real/imag parts or noting that the sum is real.
    # However, the interface expects U, V, W to have integer/half-integer entries.
    # Complex entries are not allowed.
    
    # We can split each complex tensor into real and imaginary parts.
    # This doubles the rank.
    # Rank becomes 2 * m * n * p = 72. This is worse.
    
    # Alternatively, use a real DFT?
    # Or just use the schoolbook decomposition which is real and rank 36.
    
    # Given the constraint of integer/half-integer entries, and the DFT producing complex numbers,
    # the DFT approach is difficult to implement without increasing rank or violating entry constraints.
    
    # I will fallback to the schoolbook decomposition which is valid and real.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
