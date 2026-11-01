# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen decomposition for 2x2 * 2x2
    # We apply this to two blocks of B and C.
    # Block 0: k in {0, 1}
    # Block 1: k in {2, 3}
    
    def add_strassen(k_offset, d_start):
        # k_offset is 0 or 2
        # Indices in B (m*p = 8):
        # B[0, k] -> index 0*4 + k = k
        # B[1, k] -> index 1*4 + k = 4 + k
        # Indices in C (p*n = 8):
        # C[i, k] -> index k*n + i = 2*k + i
        
        # A indices (n*m = 4):
        # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        
        # B indices for this block:
        b00 = k_offset      # B[0, k_offset]
        b01 = k_offset + 1  # B[0, k_offset+1]
        b10 = 4 + k_offset      # B[1, k_offset]
        b11 = 4 + k_offset + 1  # B[1, k_offset+1]
        
        # C indices for this block:
        c00 = 2 * k_offset      # C[0, k_offset]
        c01 = 2 * (k_offset + 1) # C[0, k_offset+1]
        c10 = 2 * k_offset + 1    # C[1, k_offset]
        c11 = 2 * (k_offset + 1) + 1 # C[1, k_offset+1]
        
        # Strassen terms
        # M1 = (A00 + A11)(B00 + B11)
        d = d_start
        U[0, d] += 1; U[3, d] += 1
        V[b00, d] += 1; V[b11, d] += 1
        W[c00, d] += 1; W[c11, d] += 1
        d += 1
        
        # M2 = (A10 + A11)B00
        U[2, d] += 1; U[3, d] += 1
        V[b00, d] += 1
        W[c10, d] += 1; W[c11, d] -= 1
        d += 1
        
        # M3 = A00(B01 - B11)
        U[0, d] += 1
        V[b01, d] += 1; V[b11, d] -= 1
        W[c01, d] += 1; W[c11, d] += 1
        d += 1
        
        # M4 = A11(B10 - B00)
        U[3, d] += 1
        V[b10, d] += 1; V[b00, d] -= 1
        W[c00, d] += 1; W[c10, d] += 1
        d += 1
        
        # M5 = (A00 + A01)B11
        U[0, d] += 1; U[1, d] += 1
        V[b11, d] += 1
        W[c00, d] -= 1; W[c01, d] += 1
        d += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2, d] += 1; U[0, d] -= 1
        V[b00, d] += 1; V[b01, d] += 1
        W[c11, d] += 1
        d += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, d] += 1; U[3, d] -= 1
        V[b10, d] += 1; V[b11, d] += 1
        W[c00, d] += 1
        d += 1
        
        return d
    
    d = add_strassen(0, 0)
    d = add_strassen(2, d)
    
    # Verify
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
