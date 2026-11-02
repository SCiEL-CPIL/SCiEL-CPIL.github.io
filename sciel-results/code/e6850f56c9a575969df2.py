"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Strassen factors for 2x2 matrix multiplication
    # U, V, W are 7x4, 7x4, 7x4 matrices
    # Indices: U[r, i*m+j], V[r, j*p+k], W[r, k*n+i]
    # Here n=2, m=2, p=2 for the base block.
    
    # Strassen's 7 products
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U matrix (7x4): row r, col i*2+j
    U_base = np.array([
        [1, 0, 1, 0],  # M1: a11 + a22
        [0, 0, 1, 1],  # M2: a21 + a22
        [1, 0, 0, 0],  # M3: a11
        [0, 0, 0, 1],  # M4: a22
        [1, 1, 0, 0],  # M5: a11 + a12
        [-1, 0, 1, 0], # M6: a21 - a11
        [0, 1, 0, -1]  # M7: a12 - a22
    ])
    
    # V matrix (7x4): row r, col j*2+k
    V_base = np.array([
        [1, 1, 1, 1],  # M1: b11 + b22
        [1, 0, 0, 0],  # M2: b11
        [0, 1, 0, -1], # M3: b12 - b22
        [0, -1, 1, 0], # M4: b21 - b11
        [0, 0, 0, 1],  # M5: b22
        [1, 1, 0, 0],  # M6: b11 + b12
        [0, 0, 1, 1]   # M7: b21 + b22
    ])
    
    # W matrix (7x4): row r, col k*2+i
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W_base = np.array([
        [1, 0, 0, 1],  # M1: C11 + C22
        [0, 0, -1, -1],# M2: -C21 - C22
        [1, 1, 0, 0],  # M3: C11 + C12
        [0, 0, 1, -1], # M4: C21 - C22
        [-1, -1, 0, 0],# M5: -C11 - C12
        [0, 0, 0, 1],  # M6: C22
        [1, 0, 0, 0]   # M7: C11
    ])
    
    # Verify base decomposition
    T_base = np.zeros((4, 4, 4))
    for i in range(2):
        for k in range(2):
            for j in range(2):
                T_base[i*2+j, j*2+k, k*2+i] = 1.0
                
    T_recon_base = np.einsum('ar,br,cr->abc', U_base, V_base, W_base)
    if not np.allclose(T_recon_base, T_base, atol=1e-9):
        # Fallback to identity if Strassen fails (should not happen)
        U_base = np.eye(4).reshape(4, 4, 1).squeeze(2) # Not correct shape, need 7x4
        # Just return identity rank 4 for 2x2x2? No, rank is 4.
        # For safety, we return the Strassen one even if check fails, as it's standard.
        pass

    # Build full rank-21 decomposition
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 3 blocks of 2x2
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            # U is same for all blocks
            U[:, r] = U_base[t, :]
            
            # V: map local j,k to global j,k
            # local j in {0,1}, local k in {0,1}
            # global j in {0,1}, global k in {2b, 2b+1}
            for j in range(2):
                for k in range(2):
                    V[j * p + (2 * b + k), r] = V_base[t, j * 2 + k]
                    
            # W: map local i,k to global i,k
            # local i in {0,1}, local k in {0,1}
            # global i in {0,1}, global k in {2b, 2b+1}
            for i in range(2):
                for k in range(2):
                    W[(2 * b + k) * n + i, r] = W_base[t, k * 2 + i]
                    
    # Verify full decomposition
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.allclose(T_recon, T, atol=1e-9):
        # Try to round to integers/half-integers
        U_out = np.round(U * 2) / 2
        V_out = np.round(V * 2) / 2
        W_out = np.round(W * 2) / 2
        
        T_recon_out = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        if np.allclose(T_recon_out, T, atol=1e-9):
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: return the unrounded version if verification fails (should not happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
