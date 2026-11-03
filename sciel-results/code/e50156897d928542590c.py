import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen 2x2 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker product
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Pre-computed permutation for 16 indices
    # Maps standard row-major index S to Kronecker-interleaved index K
    # S = R*4 + C
    # R = 2*r_out + r_in
    # C = 2*c_out + c_in
    # K = (r_out*2 + c_out)*4 + (r_in*2 + c_in)
    perm = np.array([
        # S=0: R=0, C=0 -> r_out=0, r_in=0, c_out=0, c_in=0 -> K=0*4+0=0
        0, 
        # S=1: R=0, C=1 -> r_out=0, r_in=0, c_out=0, c_in=1 -> K=0*4+1=1
        1, 
        # S=2: R=0, C=2 -> r_out=0, r_in=0, c_out=1, c_in=0 -> K=1*4+0=4
        4, 
        # S=3: R=0, C=3 -> r_out=0, r_in=0, c_out=1, c_in=1 -> K=1*4+1=5
        5, 
        # S=4: R=1, C=0 -> r_out=0, r_in=1, c_out=0, c_in=0 -> K=0*4+2=2
        2, 
        # S=5: R=1, C=1 -> r_out=0, r_in=1, c_out=0, c_in=1 -> K=0*4+3=3
        3, 
        # S=6: R=1, C=2 -> r_out=0, r_in=1, c_out=1, c_in=0 -> K=1*4+2=6
        6, 
        # S=7: R=1, C=3 -> r_out=0, r_in=1, c_out=1, c_in=1 -> K=1*4+3=7
        7, 
        # S=8: R=2, C=0 -> r_out=1, r_in=0, c_out=0, c_in=0 -> K=2*4+0=8
        8, 
        # S=9: R=2, C=1 -> r_out=1, r_in=0, c_out=0, c_in=1 -> K=2*4+1=9
        9, 
        # S=10: R=2, C=2 -> r_out=1, r_in=0, c_out=1, c_in=0 -> K=3*4+0=12
        12, 
        # S=11: R=2, C=3 -> r_out=1, r_in=0, c_out=1, c_in=1 -> K=3*4+1=13
        13, 
        # S=12: R=3, C=0 -> r_out=1, r_in=1, c_out=0, c_in=0 -> K=2*4+2=10
        10, 
        # S=13: R=3, C=1 -> r_out=1, r_in=1, c_out=0, c_in=1 -> K=2*4+3=11
        11, 
        # S=14: R=3, C=2 -> r_out=1, r_in=1, c_out=1, c_in=0 -> K=3*4+2=14
        14, 
        # S=15: R=3, C=3 -> r_out=1, r_in=1, c_out=1, c_in=1 -> K=3*4+3=15
        15
    ], dtype=int)
    
    # Apply permutation to rows
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Convert to lists
    U_list = U4_fixed.tolist()
    V_list = V4_fixed.tolist()
    W_list = W4_fixed.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
