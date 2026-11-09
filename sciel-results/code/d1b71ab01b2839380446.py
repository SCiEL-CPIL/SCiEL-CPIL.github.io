import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49  # 7 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen 2x2 coefficients
    # u2[i, j, s] for A_ij
    # v2[j, k, s] for B_jk
    # w2[i, k, s] for C_ik
    u2 = np.zeros((2, 2, 7))
    v2 = np.zeros((2, 2, 7))
    w2 = np.zeros((2, 2, 7))
    
    # M1 = (A00+A11)(B00+B11)
    u2[0,0,0] = 1; u2[1,1,0] = 1
    v2[0,0,0] = 1; v2[1,1,0] = 1
    w2[0,0,0] = 1; w2[0,1,0] = 1; w2[1,0,0] = 1; w2[1,1,0] = 1 # Wait, w2 is coeff for C_ik
    # C00 = M1 + ...
    # C01 = ...
    # C10 = ...
    # C11 = ...
    
    # Let's use standard Strassen formulas
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    u2[0,0,0] = 1; u2[1,1,0] = 1
    v2[0,0,0] = 1; v2[1,1,0] = 1
    w2[0,0,0] = 1; w2[0,1,0] = 0; w2[1,0,0] = 0; w2[1,1,0] = 1 # C00 gets M1, C11 gets M1
    
    u2[1,0,1] = 1; u2[1,1,1] = 1
    v2[0,0,1] = 1
    w2[1,0,1] = 1; w2[1,1,1] = -1 # C10 gets M2, C11 gets -M2
    
    u2[0,0,2] = 1
    v2[0,1,2] = 1; v2[1,1,2] = -1
    w2[0,1,2] = 1; w2[1,1,2] = 1 # C01 gets M3, C11 gets M3
    
    u2[1,1,3] = 1
    v2[1,0,3] = 1; v2[0,0,3] = -1
    w2[0,0,3] = 1; w2[1,0,3] = 1 # C00 gets M4, C10 gets M4
    
    u2[0,0,4] = 1; u2[0,1,4] = 1
    v2[1,1,4] = 1
    w2[0,0,4] = -1; w2[0,1,4] = 1 # C00 gets -M5, C01 gets M5
    
    u2[1,0,5] = 1; u2[0,0,5] = -1
    v2[0,0,5] = 1; v2[0,1,5] = 1
    w2[1,1,5] = 1 # C11 gets M6
    
    u2[0,1,6] = 1; u2[1,1,6] = -1
    v2[1,0,6] = 1; v2[1,1,6] = 1
    w2[0,0,6] = 1 # C00 gets M7
    
    # Outer Strassen 4x4 coefficients
    # p[t, I, J] for A_IJ
    # q[t, J, K] for B_JK
    # w4[t, K, I] for C_KI (Note: C_KI = sum J A_IJ B_JK)
    # Standard Strassen computes C_IK = sum J A_IJ B_JK
    # Here we want C_KI.
    # So we swap I and K in the output?
    # C_KI = (AB)_IK.
    # So w4[t, K, I] should be coeff of M_t in C_IK?
    # Let's check: C_IK = sum_t w_std[t, I, K] M_t.
    # We want C_KI.
    # So w4[t, K, I] = w_std[t, I, K].
    
    p = np.zeros((7, 2, 2))
    q = np.zeros((7, 2, 2))
    w4 = np.zeros((7, 2, 2)) # w4[t, K, I]
    
    # t=0 (M1)
    p[0,0,0] = 1; p[0,1,1] = 1
    q[0,0,0] = 1; q[0,1,1] = 1
    w4[0,0,0] = 1; w4[0,1,0] = 1; w4[0,0,1] = 1; w4[0,1,1] = 1 # C00, C10, C01, C11?
    # w_std[0, I, K]:
    # C00 = M1 + ... -> w_std[0,0,0]=1
    # C01 = ... -> w_std[0,0,1]=0
    # C10 = ... -> w_std[0,1,0]=0
    # C11 = M1 - ... -> w_std[0,1,1]=1
    # So w4[0, K, I] = w_std[0, I, K].
    # w4[0, 0, 0] = w_std[0, 0, 0] = 1
    # w4[0, 1, 0] = w_std[0, 0, 1] = 0
    # w4[0, 0, 1] = w_std[0, 1, 0] = 0
    # w4[0, 1, 1] = w_std[0, 1, 1] = 1
    
    w4[0,0,0] = 1
    w4[0,1,0] = 0
    w4[0,0,1] = 0
    w4[0,1,1] = 1
    
    # t=1 (M2)
    p[1,1,0] = 1; p[1,1,1] = 1
    q[1,0,0] = 1
    # C10 = M2 + ... -> w_std[1,1,0]=1
    # C11 = -M2 + ... -> w_std[1,1,1]=-1
    w4[1,0,1] = 1 # K=0, I=1 -> w_std[1,1,0]
    w4[1,1,1] = -1 # K=1, I=1 -> w_std[1,1,1]
    
    # t=2 (M3)
    p[2,0,0] = 1
    q[2,0,1] = 1; q[2,1,1] = -1
    # C01 = M3 + ... -> w_std[2,0,1]=1
    # C11 = M3 + ... -> w_std[2,1,1]=1
    w4[2,1,0] = 1 # K=1, I=0 -> w_std[2,0,1]
    w4[2,1,1] = 1 # K=1, I=1 -> w_std[2,1,1]
    
    # t=3 (M4)
    p[3,1,1] = 1
    q[3,1,0] = 1; q[3,0,0] = -1
    # C00 = M4 + ... -> w_std[3,0,0]=1
    # C10 = M4 + ... -> w_std[3,1,0]=1
    w4[3,0,0] = 1 # K=0, I=0 -> w_std[3,0,0]
    w4[3,0,1] = 1 # K=0, I=1 -> w_std[3,1,0]
    
    # t=4 (M5)
    p[4,0,0] = 1; p[4,0,1] = 1
    q[4,1,1] = 1
    # C00 = -M5 + ... -> w_std[4,0,0]=-1
    # C01 = M5 + ... -> w_std[4,0,1]=1
    w4[4,0,0] = -1 # K=0, I=0 -> w_std[4,0,0]
    w4[4,1,0] = 1 # K=1, I=0 -> w_std[4,0,1]
    
    # t=5 (M6)
    p[5,1,0] = 1; p[5,0,0] = -1
    q[5,0,0] = 1; q[5,0,1] = 1
    # C11 = M6 + ... -> w_std[5,1,1]=1
    w4[5,1,1] = 1 # K=1, I=1 -> w_std[5,1,1]
    
    # t=6 (M7)
    p[6,0,1] = 1; p[6,1,1] = -1
    q[6,1,0] = 1; q[6,1,1] = 1
    # C00 = M7 + ... -> w_std[6,0,0]=1
    w4[6,0,0] = 1 # K=0, I=0 -> w_std[6,0,0]
    
    # Construct U, V, W
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            for I in range(2):
                for J in range(2):
                    if p[t, I, J] != 0:
                        for ip in range(2):
                            for jp in range(2):
                                if u2[ip, jp, s] != 0:
                                    val = p[t, I, J] * u2[ip, jp, s]
                                    rA = (I * 2 + ip) * 4 + (J * 2 + jp)
                                    U[rA][r] += val
                                    
            for J in range(2):
                for K in range(2):
                    if q[t, J, K] != 0:
                        for jp in range(2):
                            for kp in range(2):
                                if v2[jp, kp, s] != 0:
                                    val = q[t, J, K] * v2[jp, kp, s]
                                    rB = (J * 2 + jp) * 4 + (K * 2 + kp)
                                    V[rB][r] += val
                                    
            for K in range(2):
                for I in range(2):
                    if w4[t, K, I] != 0:
                        for kp in range(2):
                            for ip in range(2):
                                if w2[ip, kp, s] != 0:
                                    val = w4[t, K, I] * w2[ip, kp, s]
                                    rC = (K * 2 + kp) * 4 + (I * 2 + ip)
                                    W[rC][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
