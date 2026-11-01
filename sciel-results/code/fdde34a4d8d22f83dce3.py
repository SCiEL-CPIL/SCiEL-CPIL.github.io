import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 140
    
    # Rank 56 construction using Strassen on 8 blocks
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Output:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # We define the factors for each Strassen step s in 0..6
    # x_factors: list of (i, j, coeff) for A
    # y_factors: list of (j, k, coeff) for B
    # w_coeffs: dict (i, k) -> coeff for C
    
    # Note: In the global U, V, W, the indices are:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # For a specific block:
    # A_block is A[i, 2*alpha + j_local]
    # B_block is B[2*beta + j_local, 2*gamma + k_local]
    # C_block is C[i, 2*gamma + k_local]
    
    # Strassen Step 0: M1
    # X = A00 + A11 -> (0,0,1), (1,1,1)
    # Y = B00 + B11 -> (0,0,1), (1,1,1)
    # W: C00+=1, C11+=1
    steps = [
        {
            'x': [(0,0,1), (1,1,1)],
            'y': [(0,0,1), (1,1,1)],
            'w': [(0,0,1), (1,1,1)]
        },
        {
            'x': [(1,0,1), (1,1,1)],
            'y': [(0,0,1)],
            'w': [(1,0,1), (1,1,-1)]
        },
        {
            'x': [(0,0,1)],
            'y': [(1,0,1), (1,1,-1)],
            'w': [(0,1,1), (1,1,1)]
        },
        {
            'x': [(1,1,1)],
            'y': [(0,1,1), (0,0,-1)],
            'w': [(0,0,1), (1,0,1)]
        },
        {
            'x': [(0,0,1), (0,1,1)],
            'y': [(1,1,1)],
            'w': [(0,0,-1), (0,1,1)]
        },
        {
            'x': [(1,0,1), (0,0,-1)],
            'y': [(0,0,1), (0,1,1)],
            'w': [(1,1,1)]
        },
        {
            'x': [(0,1,1), (1,1,-1)],
            'y': [(1,0,1), (1,1,1)],
            'w': [(0,0,1)]
        }
    ]
    
    r = 0
    for alpha in range(2): # A col block 0, 1
        for beta in range(2): # B row block 0, 1
            if alpha != beta: continue # Only matching blocks contribute to C
            for gamma in range(4): # B col block 0, 1, 2, 3
                for s in range(7):
                    step = steps[s]
                    # Fill U
                    for (i_loc, j_loc, coeff) in step['x']:
                        i_global = i_loc
                        j_global = 2 * alpha + j_loc
                        U[i_global * m + j_global, r] = coeff
                    
                    # Fill V
                    for (j_loc, k_loc, coeff) in step['y']:
                        j_global = 2 * beta + j_loc
                        k_global = 2 * gamma + k_loc
                        V[j_global * p + k_global, r] = coeff
                    
                    # Fill W
                    for (i_loc, k_loc, coeff) in step['w']:
                        i_global = i_loc
                        k_global = 2 * gamma + k_loc
                        W[k_global * n + i_global, r] = coeff
                    
                    r += 1
    
    # Verify
    def verify(U, V, W):
        rng = np.random.RandomState(42)
        for _ in range(10):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            U_r = U.reshape(n, m, -1)
            V_r = V.reshape(m, p, -1)
            W_r = W.reshape(p, n, -1)
            X = np.einsum('ijr,ij->ir', U_r, A)
            Y = np.einsum('jkr,jk->kr', V_r, B)
            C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_r)
            if not np.allclose(C_recon, C_true, atol=1e-5):
                return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
