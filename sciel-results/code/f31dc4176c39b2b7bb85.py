import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        
        for (i, j), val in u_dict.items():
            u_vec[i*m + j] = val
        for (j, k), val in v_dict.items():
            v_vec[j*p + k] = val
        for (k, i), val in w_dict.items():
            w_vec[k*n + i] = val
            
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # 1. 4x4 Block (indices 0..3) using Strassen (Rank 49)
    # We define the 49 terms for 4x4 Strassen.
    # This is derived from 7 2x2 Strassen multiplications.
    
    # 2x2 Strassen terms (local indices 0,1)
    # M1: u={(0,0):1,(1,1):1}, v={(0,0):1,(1,1):1}
    # M2: u={(1,0):1,(1,1):1}, v={(0,0):1}
    # M3: u={(0,0):1}, v={(1,0):1,(1,1):-1}
    # M4: u={(1,1):1}, v={(0,1):1,(0,0):-1}
    # M5: u={(0,0):1,(0,1):1}, v={(1,1):1}
    # M6: u={(1,0):1,(0,0):-1}, v={(0,0):1,(0,1):1}
    # M7: u={(0,1):1,(1,1):-1}, v={(1,0):1,(1,1):1}
    
    # For 4x4, we have blocks A11, A12, A21, A22 and B11, B12, B21, B22.
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each P is a 2x2 matrix.
    # We need to expand each P into 7 scalar terms.
    # And then map to C.
    
    # This is complex. I will hardcode the 49 terms for the 4x4 block.
    # Or, I can use a library if available? No.
    
    # Let's just use the schoolbook for the 4x4 block (Rank 64) 
    # and see if I can optimize the cross terms?
    # No, cross terms are minimal.
    
    # I will implement the 49 terms manually in the code below.
    # It's tedious but doable.
    
    # Actually, I can just return the schoolbook if I can't guarantee the Strassen implementation.
    # But 1.0 is the baseline. I want > 1.0.
    
    # I will implement the 4x4 Strassen.
    
    pass
