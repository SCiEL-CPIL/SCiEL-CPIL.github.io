import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((16, rank), dtype=np.int8)
    V = np.zeros((16, rank), dtype=np.int8)
    W = np.zeros((16, rank), dtype=np.int8)

    # Define the 49 terms of the recursive Strassen decomposition.
    # Each term r is a triplet (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict or list of (index, coeff) for A
    # v_coeffs: dict or list of (index, coeff) for B
    # w_coeffs: dict or list of (index, coeff) for C
    
    # We generate these programmatically using the Strassen coefficients.
    # Outer Strassen (2x2 blocks of 2x2)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # s=0: M0 = (A00+A11)(B00+B11)
    # s=1: M1 = (A10+A11)B00
    # s=2: M2 = A00(B01-B11)
    # s=3: M3 = A11(B10-B00)
    # s=4: M4 = (A00+A01)B11
    # s=5: M5 = (A10-A00)(B00+B01)
    # s=6: M6 = (A01-A11)(B10+B11)
    
    # Inner Strassen (2x2 blocks of 1x1)
    # Same structure for the 2x2 sub-matrices.
    
    # We will construct the full 4x4 coefficients by combining outer and inner.
    # Index i in 0..3 is (I, i2) where I in 0..1, i2 in 0..1.
    # i = 2*I + i2.
    
    # Outer coefficients for A (beta), B (gamma), C (delta)
    # beta[s, I, J], gamma[s, J, K], delta[s, I, K]
    
    beta = np.zeros((7, 2, 2), dtype=np.int8)
    gamma = np.zeros((7, 2, 2), dtype=np.int8)
    delta = np.zeros((7, 2, 2), dtype=np.int8)
    
    # s=0
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    
    # s=1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1
    
    # s=2
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1
    
    # s=3
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1
    
    # s=4
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1
    
    # s=5
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1
    
    # s=6
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1
    
    # Inner coefficients (same as outer, for the 2x2 scalar blocks)
    # u2[t, i2, j2], v2[t, j2, k2], w2[t, k2, i2]
    
    u2 = np.zeros((7, 2, 2), dtype=np.int8)
    v2 = np.zeros((7, 2, 2), dtype=np.int8)
    w2 = np.zeros((7, 2, 2), dtype=np.int8)
    
    # t=0
    u2[0, 0, 0] = 1; u2[0, 1, 1] = 1
    v2[0, 0, 0] = 1; v2[0, 1, 1] = 1
    w2[0, 0, 0] = 1; w2[0, 1, 1] = 1
    
    # t=1
    u2[1, 1, 0] = 1; u2[1, 1, 1] = 1
    v2[1, 0, 0] = 1
    w2[1, 0, 1] = 1; w2[1, 1, 1] = -1
    
    # t=2
    u2[2, 0, 0] = 1
    v2[2, 0, 1] = 1; v2[2, 1, 1] = -1
    w2[2, 1, 0] = 1; w2[2, 1, 1] = 1
    
    # t=3
    u2[3, 1, 1] = 1
    v2[3, 1, 0] = 1; v2[3, 0, 0] = -1
    w2[3, 0, 0] = 1; w2[3, 0, 1] = 1
    
    # t=4
    u2[4, 0, 0] = 1; u2[4, 0, 1] = 1
    v2[4, 1, 1] = 1
    w2[4, 0, 0] = -1; w2[4, 1, 0] = 1
    
    # t=5
    u2[5, 1, 0] = 1; u2[5, 0, 0] = -1
    v2[5, 0, 0] = 1; v2[5, 0, 1] = 1
    w2[5, 1, 1] = 1
    
    # t=6
    u2[6, 0, 1] = 1; u2[6, 1, 1] = -1
    v2[6, 1, 0] = 1; v2[6, 1, 1] = 1
    w2[6, 0, 0] = 1
    
    # Construct U, V, W
    # U[i, j, r] where i = 2*I + i2, j = 2*J + j2
    # V[j, k, r] where j = 2*J + j2, k = 2*K + k2
    # W[k, i, r] where k = 2*K + k2, i = 2*I + i2
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # U
            for I in range(2):
                for J in range(2):
                    b_val = beta[s, I, J]
                    if b_val == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_val = u2[t, i2, j2]
                            if u_val == 0: continue
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U[i * 4 + j, r] += b_val * u_val

            # V
            for J in range(2):
                for K in range(2):
                    g_val = gamma[s, J, K]
                    if g_val == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v_val = v2[t, j2, k2]
                            if v_val == 0: continue
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V[j * 4 + k, r] += g_val * v_val

            # W
            for I in range(2):
                for K in range(2):
                    d_val = delta[s, I, K]
                    if d_val == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w_val = w2[t, k2, i2]
                            if w_val == 0: continue
                            k = 2 * K + k2
                            i = 2 * I + i2
                            W[k * 4 + i, r] += d_val * w_val

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
