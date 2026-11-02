"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    R = 14

    U = np.zeros((nm, R), dtype=int)
    V = np.zeros((mp, R), dtype=int)
    W = np.zeros((pn, R), dtype=int)

    # Strassen block definitions for 2x2 x 2
    # U indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V indices for block col 0-1: 00,01,10,11 -> 0,1,4,5
    # V indices for block col 2-3: 00,01,10,11 -> 2,3,6,7
    # W indices for block col 0-1: 00,10,01,11 -> 0,1,2,3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    # W indices for block col 2-3: 00,10,01,11 -> 4,5,6,7 (2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7)

    # Define the 7 Strassen steps
    # (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    strassen_steps = [
        # M1: (a00+a11)(b00+b11)
        ((0, 3), (1, 1), (0, 4), (1, 1), (0, 3), (1, 1)),
        # M2: (a10+a11)b00
        ((2, 3), (1, 1), (0,), (1,), (1, 3), (1, -1)),
        # M3: a00(b01-b11)
        ((0,), (1,), (1, 4), (1, -1), (2, 3), (1, 1)),
        # M4: a11(b10-b00)
        ((3,), (1,), (4, 0), (1, -1), (0, 1), (1, 1)),
        # M5: (a00+a01)b11
        ((0, 1), (1, 1), (4,), (1,), (0, 2), (-1, 1)),
        # M6: (a10-a00)(b00+b01)
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),
        # M7: (a01-a11)(b10+b11)
        ((1, 3), (1, -1), (4, 5), (1, 1), (0,), (1,))
    ]
    
    # Helper to populate U, V, W
    def populate_block(offset_r, offset_v, offset_w):
        for s_idx, (u_idx, u_c, v_idx, v_c, w_idx, w_c) in enumerate(strassen_steps):
            r = offset_r + s_idx
            for i, c in zip(u_idx, u_c):
                U[i, r] = c
            for i, c in zip(v_idx, v_c):
                V[i + offset_v, r] = c
            for i, c in zip(w_idx, w_c):
                W[i + offset_w, r] = c

    # Left block (cols 0,1 of B)
    # V offset 0 (b00=0, b01=1, b10=4, b11=5)
    # W offset 0 (c00=0, c10=1, c01=2, c11=3)
    populate_block(0, 0, 0)

    # Right block (cols 2,3 of B)
    # V offset 2 (b02=2, b03=3, b12=6, b13=7)
    #   Note: V indices for right block in strassen_steps were relative (0,1,4,5).
    #   b00_right = V[2], b01_right = V[3], b10_right = V[6], b11_right = V[7].
    #   So offset_v = 2 works for 0->2, 1->3.
    #   But 4->6, 5->7. Offset 2 works for 4+2=6, 5+2=7. Correct.
    # W offset 4 (c02=4, c12=5, c03=6, c13=7)
    #   W indices in steps: 0,1,2,3.
    #   c00_right = W[4], c10_right = W[5], c01_right = W[6], c11_right = W[7].
    #   Offset 4 works.
    populate_block(7, 2, 4)

    # Verification
    Ua = U.astype(float)
    Va = V.astype(float)
    Wa = W.astype(float)
    
    # Fast check using random vectors
    np.random.seed(42)
    x = np.random.rand(nm)
    y = np.random.rand(mp)
    z = np.random.rand(pn)
    
    # Reconstruct tensor
    T = np.zeros((n, m, p))
    for r in range(R):
        u = Ua[:, r].reshape(n, m)
        v = Va[:, r].reshape(m, p)
        w = Wa[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
        
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to naive 16
        R_naive = n * m * p
        Un = np.zeros((nm, R_naive), dtype=int)
        Vn = np.zeros((mp, R_naive), dtype=int)
        Wn = np.zeros((pn, R_naive), dtype=int)
        r_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Un[i*m+j, r_idx] = 1
                    Vn[j*p+k, r_idx] = 1
                    Wn[k*n+i, r_idx] = 1
                    r_idx += 1
        return {"U": Un.tolist(), "V": Vn.tolist(), "W": Wn.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
