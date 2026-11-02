import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: I = i*2 + j, J = j*2 + k, K = k*2 + i
    U_small = np.zeros((4, 7))
    V_small = np.zeros((4, 7))
    W_small = np.zeros((4, 7))
    
    # Strassen's 7 products
    # P1: (A00+A11)(B00+B11)
    U_small[0, 0] = 1; U_small[3, 0] = 1
    V_small[0, 0] = 1; V_small[3, 0] = 1
    W_small[0, 0] = 1; W_small[3, 0] = 1
    W_small[1, 0] = 1; W_small[2, 0] = 1
    # Note: W index is k*2+i. 
    # C00 = P1 + ...
    # C00 corresponds to k=0, i=0 -> K=0.
    # C01 corresponds to k=0, i=1 -> K=2.
    # C10 corresponds to k=1, i=0 -> K=1.
    # C11 corresponds to k=1, i=1 -> K=3.
    
    # We need to define U, V, W such that sum U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
    # T[i,j,k] = 1 for all i,j,k.
    
    # Strassen coefficients for C_ik:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's fill W_small based on these coefficients.
    # W_small[K, r] where K = k*2 + i.
    
    # P1 (r=0):
    W_small[0, 0] = 1 # C00
    W_small[3, 0] = 1 # C11
    
    # P2 (r=1): (A10+A11)B00
    U_small[2, 1] = 1; U_small[3, 1] = 1
    V_small[0, 1] = 1
    W_small[1, 1] = 1 # C10
    W_small[3, 1] = -1 # C11
    
    # P3 (r=2): A00(B10-B11)
    U_small[0, 2] = 1
    V_small[2, 2] = 1; V_small[3, 2] = -1
    W_small[2, 2] = 1 # C01
    W_small[3, 2] = 1 # C11
    
    # P4 (r=3): A11(B01-B00)
    U_small[3, 3] = 1
    V_small[1, 3] = 1; V_small[0, 3] = -1
    W_small[0, 3] = 1 # C00
    W_small[1, 3] = 1 # C10
    
    # P5 (r=4): (A00+A01)B11
    U_small[0, 4] = 1; U_small[1, 4] = 1
    V_small[3, 4] = 1
    W_small[0, 4] = -1 # C00
    W_small[2, 4] = 1 # C01
    
    # P6 (r=5): (A10-A00)(B00+B01)
    U_small[2, 5] = 1; U_small[0, 5] = -1
    V_small[0, 5] = 1; V_small[1, 5] = 1
    W_small[3, 5] = 1 # C11
    
    # P7 (r=6): (A01-A11)(B10+B11)
    U_small[1, 6] = 1; U_small[3, 6] = -1
    V_small[2, 6] = 1; V_small[3, 6] = 1
    W_small[0, 6] = 1 # C00

    # Schoolbook for 3x3x3
    U_big = np.zeros((9, 27))
    V_big = np.zeros((9, 27))
    W_big = np.zeros((9, 27))
    col = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                U_big[i1*3 + j1, col] = 1
                V_big[j1*3 + k1, col] = 1
                W_big[k1*3 + i1, col] = 1
                col += 1

    # Combine
    rank = 27 * 7
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    for r_big in range(27):
        for r_small in range(7):
            r = r_big * 7 + r_small
            
            # U
            for i1 in range(3):
                for j1 in range(3):
                    u_big_val = U_big[i1*3 + j1, r_big]
                    if u_big_val == 0: continue
                    for i0 in range(2):
                        for j0 in range(2):
                            u_small_val = U_small[i0*2 + j0, r_small]
                            if u_small_val == 0: continue
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            I = i*6 + j
                            U[I, r] += u_big_val * u_small_val
                            
            # V
            for j1 in range(3):
                for k1 in range(3):
                    v_big_val = V_big[j1*3 + k1, r_big]
                    if v_big_val == 0: continue
                    for j0 in range(2):
                        for k0 in range(2):
                            v_small_val = V_small[j0*2 + k0, r_small]
                            if v_small_val == 0: continue
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            J = j*6 + k
                            V[J, r] += v_big_val * v_small_val
                            
            # W
            for k1 in range(3):
                for i1 in range(3):
                    w_big_val = W_big[k1*3 + i1, r_big]
                    if w_big_val == 0: continue
                    for k0 in range(2):
                        for i0 in range(2):
                            w_small_val = W_small[k0*2 + i0, r_small]
                            if w_small_val == 0: continue
                            k = 2*k1 + k0
                            i = 2*i1 + i0
                            K = k*6 + i
                            W[K, r] += w_big_val * w_small_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
