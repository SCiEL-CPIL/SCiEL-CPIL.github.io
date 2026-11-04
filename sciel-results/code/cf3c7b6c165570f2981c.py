import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n * m  # 15
    
    U = np.zeros((n * m, rank), dtype=complex)
    V = np.zeros((m * p, rank), dtype=complex)
    W = np.zeros((p * n, rank), dtype=complex)
    
    omega = np.exp(2j * np.pi / m)
    
    # Precompute powers of omega
    # pow_omega[j, r] = omega^(j*r)
    pow_omega = np.zeros((m, m), dtype=complex)
    for j in range(m):
        for r in range(m):
            pow_omega[j, r] = omega ** (j * r)
            
    for i in range(n):
        for r in range(m):
            c = i * m + r
            
            # U: rows i*m .. i*m+m-1
            # U[i*m + j, c] = omega^(j*r)
            for j in range(m):
                U[i * m + j, c] = pow_omega[j, r]
                
            # V: rows j*p .. j*p+p-1
            # V[j*p + k, c] = omega^(-j*r)
            # This is independent of i, so we could fill it once, but it's fine here.
            for j in range(m):
                val = pow_omega[j, r].conjugate() # omega^(-jr)
                for k in range(p):
                    V[j * p + k, c] = val
                    
            # W: rows k*n + i
            # W[k*n + i, c] = 1/m
            for k in range(p):
                W[k * n + i, c] = 1.0 / m

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
