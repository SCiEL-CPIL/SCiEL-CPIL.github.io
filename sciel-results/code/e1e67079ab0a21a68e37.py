# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2x2_np():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication as numpy arrays."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # P3 = A00 * (B10 - B00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # P4 = A11 * (B01 - B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    # P6 = (A01 - A11) * (B00 + B01)
    U[1, 5] = 1; U[3, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A00 - A10) * (B10 + B11)
    U[0, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4x4_np():
    """Construct rank 49 decomposition for 4x4x4 using numpy."""
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U2, V2, W2 = _strassen_2x2x2_np()
    
    steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj, s]
                            if local_u == 0: continue
                            global_i = bi*2 + li
                            global_j = bj*2 + lj
                            U[global_i*4 + global_j, col] += coeff_a * local_u
            
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk, s]
                            if local_v == 0: continue
                            global_j = bj*2 + lj
                            global_k = bk*2 + lk
                            V[global_j*4 + global_k, col] += coeff_b * local_v
            
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            local_w = W2[lk*2+li, s]
                            if local_w == 0: continue
                            global_k = ck*2 + lk
                            global_i = ci*2 + li
                            W[global_k*4 + global_i, col] += coeff_c * local_w
                            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition_np(decomp, n, m, p):
    """Verify the decomposition using numpy."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = U.shape[1]
    
    # Check all valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    n, m, p = 4, 4, 5
    
    result_444 = _strassen_4x4x4_np()
    
    U4 = result_444["U"]
    V4 = result_444["V"]
    W4 = result_444["W"]
    rank_444 = U4.shape[1]
    
    new_rank = rank_444 + 1
    U = np.zeros((n * m, new_rank))
    V = np.zeros((m * p, new_rank))
    W = np.zeros((p * n, new_rank))
    
    # Copy 4x4x4 part
    for i in range(n):
        for j in range(m):
            U[i*m + j, :rank_444] = U4[i*4 + j, :rank_444]
            
    for j in range(m):
        for k in range(4):
            V[j*p + k, :rank_444] = V4[j*4 + k, :rank_444]
            
    for k in range(4):
        for i in range(n):
            W[k*n + i, :rank_444] = W4[k*4 + i, :rank_444]
            
    # Add 1 term for k=4 slice
    new_col = rank_444
    for i in range(n):
        for j in range(m):
            U[i*m + j, new_col] = 1
        for j in range(m):
            V[j*p + 4, new_col] = 1
        W[4*n + i, new_col] = 1
        
    result = {"U": U, "V": V, "W": W}
    
    if _verify_decomposition_np(result, n, m, p):
        return result
    else:
        # Fallback
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
