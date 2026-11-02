import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 14 decomposition: two independent Strassen 2x2x2 blocks
    rank = 14
    U = np.zeros((n * m, rank))  # (4, 14)
    V = np.zeros((m * p, rank))  # (8, 14)
    W = np.zeros((p * n, rank))  # (8, 14)
    
    # Index conventions:
    # A[i] where i = row*m + col  (0-indexed)
    #   A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B[j] where j = row*p + col  (0-indexed)
    #   B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03,
    #   B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C[k] where k = col*n + row  (trace convention W[k*n+i])
    #   C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11,
    #   C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    def add_strassen_block(col_offset, d_start):
        """Add a Strassen 2x2x2 decomposition for B columns [col_offset, col_offset+1]."""
        # B indices for this 2x2 block
        v11 = col_offset          # B[0, col_offset]
        v12 = col_offset + 1      # B[0, col_offset+1]
        v21 = 4 + col_offset      # B[1, col_offset]
        v22 = 4 + col_offset + 1  # B[1, col_offset+1]
        
        # C indices for output columns col_offset, col_offset+1
        # C[row, col] -> index = col * n + row
        w00 = col_offset * 2      # C[0, col_offset]
        w01 = (col_offset + 1) * 2  # C[0, col_offset+1]
        w10 = col_offset * 2 + 1    # C[1, col_offset]
        w11 = (col_offset + 1) * 2 + 1  # C[1, col_offset+1]
        
        # A indices
        a00 = 0; a01 = 1; a10 = 2; a11 = 3
        
        d = d_start
        
        # M1 = (A00 + A11)(B11 + B22)
        U[a00, d] += 1; U[a11, d] += 1
        V[v11, d] += 1; V[v22, d] += 1
        W[w00, d] += 1; W[w11, d] += 1
        d += 1
        
        # M2 = (A10 + A11)B11
        U[a10, d] += 1; U[a11, d] += 1
        V[v11, d] += 1
        W[w10, d] += 1; W[w11, d] -= 1
        d += 1
        
        # M3 = A00(B12 - B22)
        U[a00, d] += 1
        V[v12, d] += 1; V[v22, d] -= 1
        W[w01, d] += 1; W[w11, d] += 1
        d += 1
        
        # M4 = A11(B21 - B11)
        U[a11, d] += 1
        V[v21, d] += 1; V[v11, d] -= 1
        W[w00, d] += 1; W[w10, d] += 1
        d += 1
        
        # M5 = (A00 + A01)B22
        U[a00, d] += 1; U[a01, d] += 1
        V[v22, d] += 1
        W[w00, d] -= 1; W[w01, d] += 1
        d += 1
        
        # M6 = (A10 - A00)(B11 + B12)
        U[a10, d] += 1; U[a00, d] -= 1
        V[v11, d] += 1; V[v12, d] += 1
        W[w11, d] += 1
        d += 1
        
        # M7 = (A01 - A11)(B21 + B22)
        U[a01, d] += 1; U[a11, d] -= 1
        V[v21, d] += 1; V[v22, d] += 1
        W[w00, d] += 1
        d += 1
        
        return d
    
    # Add left block (B columns 0,1)
    d = add_strassen_block(0, 0)
    # Add right block (B columns 2,3)
    d = add_strassen_block(2, d)
    
    # Verify decomposition correctness
    if time.time() - start_time < 140:
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build expected tensor: T[a_idx, b_idx, c_idx] = 1 iff c corresponds to a*b
        T_expected = np.zeros((n * m, m * p, p * n))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    a_idx = a * m + b
                    b_idx = b * p + c
                    c_idx = c * n + a
                    T_expected[a_idx, b_idx, c_idx] = 1.0
        
        if not np.allclose(T_recon, T_expected, atol=1e-10):
            # Fallback to schoolbook rank-16
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col = 0
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        a_idx = a * m + b
                        b_idx = b * p + c
                        c_idx = c * n + a
                        U_fb[a_idx, col] = 1.0
                        V_fb[b_idx, col] = 1.0
                        W_fb[c_idx, col] = 1.0
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
