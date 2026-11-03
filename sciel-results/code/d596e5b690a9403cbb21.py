import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """Rank-49 decomposition of 4x4 matrix multiplication via Kronecker product of Strassen."""
    deadline = time.time() + 100.0
    n, m, p = DIMENSIONS
    
    # Define Strassen 2x2 decomposition
    # U2[i*2+j, t], V2[j*2+k, t], W2[k*2+i, t]
    # Indices for 2x2:
    # A: a=0, b=1, c=2, d=3
    # B: e=0, f=1, g=2, h=3
    # C: p=0, r=1, q=2, s=3 (Note: W index is k*2+i)
    #    p=C[0,0] -> k=0, i=0 -> 0
    #    r=C[1,0] -> k=0, i=1 -> 1
    #    q=C[0,1] -> k=1, i=0 -> 2
    #    s=C[1,1] -> k=1, i=1 -> 3
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (a+b)(e+f)
    U2[0, 0] = 1; U2[1, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    # p = M0 - M3 + M4 - M6
    # q = M3 + M5 - M0 + M2
    # r = M0 - M1 + M2 - M5
    # s = M1 - M2 + M3 + M4
    W2[0, 0] = 1
    W2[2, 0] = -1
    W2[1, 0] = 1
    
    # Term 1: (a-b)(e+h)
    U2[0, 1] = 1; U2[1, 1] = -1
    V2[0, 1] = 1; V2[3, 1] = 1
    W2[1, 1] = -1
    W2[3, 1] = 1
    
    # Term 2: (c+d)g
    U2[2, 2] = 1; U2[3, 2] = 1
    V2[2, 2] = 1
    W2[2, 2] = 1
    W2[1, 2] = 1
    W2[3, 2] = -1
    
    # Term 3: (a+d)h
    U2[0, 3] = 1; U2[3, 3] = 1
    V2[3, 3] = 1
    W2[0, 3] = -1
    W2[2, 3] = 1
    W2[3, 3] = 1
    
    # Term 4: (a+c)e
    U2[0, 4] = 1; U2[2, 4] = 1
    V2[0, 4] = 1
    W2[0, 4] = 1
    W2[3, 4] = 1
    
    # Term 5: (b+d)(f+g-h)
    U2[1, 5] = 1; U2[3, 5] = 1
    V2[1, 5] = 1; V2[2, 5] = 1; V2[3, 5] = -1
    W2[2, 5] = 1
    W2[1, 5] = -1
    
    # Term 6: (b+c)(e-f)
    U2[1, 6] = 1; U2[2, 6] = 1
    V2[0, 6] = 1; V2[1, 6] = -1
    W2[0, 6] = -1
    
    # Construct 4x4 decomposition via Kronecker product
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r0 in range(7):
        for r1 in range(7):
            col = r0 * 7 + r1
            for i0 in range(2):
                for j0 in range(2):
                    for i1 in range(2):
                        for j1 in range(2):
                            i = 2 * i0 + i1
                            j = 2 * j0 + j1
                            u_val = U2[i0 * 2 + j0, r0] * U2[i1 * 2 + j1, r1]
                            U[i * 4 + j, col] = u_val
                            
                            # V indices: j0, k0, j1, k1
                            # V2 index: j*2+k
                            # We need to sum over k0, k1 to fill V rows?
                            # No, V is a matrix. We fill each row independently.
                            # V[j*4+k, col] = V2[j0*2+k0, r0] * V2[j1*2+k1, r1]
                            # So we iterate k0, k1 as well for V and W.
                            
            # It is more efficient to fill V and W in the same loop or separate loops.
            # Let's just do separate loops for clarity and correctness.
            
            for j0 in range(2):
                for k0 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            j = 2 * j0 + j1
                            k = 2 * k0 + k1
                            v_val = V2[j0 * 2 + k0, r0] * V2[j1 * 2 + k1, r1]
                            V[j * 4 + k, col] = v_val
                            
            for k0 in range(2):
                for i0 in range(2):
                    for k1 in range(2):
                        for i1 in range(2):
                            k = 2 * k0 + k1
                            i = 2 * i0 + i1
                            w_val = W2[k0 * 2 + i0, r0] * W2[k1 * 2 + i1, r1]
                            W[k * 4 + i, col] = w_val

    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
