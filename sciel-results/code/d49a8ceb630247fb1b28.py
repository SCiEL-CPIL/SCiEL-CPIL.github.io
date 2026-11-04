import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    
    # Ordering: r = i * (m * p) + j * p + k
    # U: rows (i, j), cols r. 1 if i,j match.
    # U = I_{n*m} @ kron(I_{n*m}, I_p)? No.
    # U is block diagonal with I_{n*m} repeated p times? No.
    # For fixed i, j, the cols are i*m*p + j*p + k.
    # This is equivalent to I_{n*m} @ (I_{n*m} kron I_p) ?
    
    # Let's just use the Kronecker product for U as derived:
    # U = I_{n*m} @ (I_{n*m} kron I_p) is wrong shape.
    # U = I_{n*m} (n*m x n*m) kron I_p (p x p) -> (n*m*p) x (n*m*p).
    # We need U to be (n*m) x (n*m*p).
    # So U = I_{n*m} @ (I_{n*m} kron I_p) ? No.
    # U_{row, col} where row in [0, n*m), col in [0, n*m*p).
    # U = I_{n*m} @ (I_{n*m} kron I_p) doesn't make sense.
    # U = I_{n*m} (as a matrix) multiplied by a selector?
    
    # Actually, U = I_{n*m} @ (I_{n*m} @ I_p) ?
    # Let's just use np.eye(n*m, R) and fill.
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
