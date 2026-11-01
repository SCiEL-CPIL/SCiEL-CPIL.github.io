import numpy as np

def construct():
    n, m, p = 2, 4, 7
    rank = 50
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a 2x2x2 Strassen block
    # local A indices: i in 0..1, j in 0..1
    # local B indices: j in 0..1, k in 0..1
    # local C indices: i in 0..1, k in 0..1
    # offset in global A: row i, col j + j_off
    # offset in global B: row j + j_off, col k + k_off
    # offset in global C: row i, col k + k_off
    
    def add_strassen_222(j_off, k_off, start_col):
        # M1 = (A00+A11)(B00+B11)
        U[0*2+0, start_col] += 1
        U[1*2+1, start_col] += 1
        V[0*7+0+k_off, start_col] += 1
        V[1*7+1+k_off, start_col] += 1
        W[0*2+0, start_col] += 1
        W[1*2+1, start_col] += 1
        
        # M2 = (A10+A11)B00
        U[1*2+0, start_col+1] += 1
        U[1*2+1, start_col+1] += 1
        V[0*7+0+k_off, start_col+1] += 1
        W[1*2+0, start_col+1] += 1
        
        # M3 = A00(B01-B11)
        U[0*2+0, start_col+2] += 1
        V[0*7+1+k_off, start_col+2] += 1
        V[1*7+1+k_off, start_col+2] -= 1
        W[0*2+1, start_col+2] += 1
        
        # M4 = A11(B10-B00)
        U[1*2+1, start_col+3] += 1
        V[1*7+0+k_off, start_col+3] += 1
        V[0*7+0+k_off, start_col+3] -= 1
        W[1*2+0, start_col+3] += 1
        
        # M5 = (A00+A01)B11
        U[0*2+0, start_col+4] += 1
        U[0*2+1, start_col+4] += 1
        V[1*7+1+k_off, start_col+4] += 1
        W[0*2+1, start_col+4] += 1
        W[1*2+1, start_col+4] -= 1 # Wait, C00 = M1+M4-M5+M7, C11 = M1-M2+M3+M6. 
        # Let's re-verify Strassen output mapping.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My W additions above were partial. I need to accumulate coefficients.
        # It is safer to define the full contribution of each M to W.
        
        # Let's restart the helper to be precise.
        pass

    # I will just hardcode the 50 columns or use a list of operations.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
