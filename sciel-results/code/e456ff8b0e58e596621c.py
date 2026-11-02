import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    nr, mr, pr = n * m, m * p, p * n  # 4, 16, 16

    # ------------------------------------------------------------------
    # The multiplication tensor: T[a,b,c] = 1 with a=i*m+j, b=j*p+k, c=k*n+i
    # (the official trace convention for W).  T has 32 ones, one per (i,j,k).
    # ------------------------------------------------------------------
    T = np.zeros((nr, mr, pr), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-8):
        Tc = np.einsum('at,bt,ct->abc', U, V, W)
        err = float(np.max(np.abs(Tc - T)))
        return err < tol, err

    # ------------------------------------------------------------------
    # Deterministic rank-28 construction: Strassen (rank-7) applied to each of the
    # 4 column-pairs (2x2 x 2x2 blocks).  This is exactly correct and minimal.
    # The rank of M_{2,2,8} is 28 because it is the direct sum of four M_{2,2,2}
    # tensors with disjoint supports in the B and C indices.
    # ------------------------------------------------------------------
    def get_strassen_28():
        rank = 28
        U = np.zeros((nr, rank), dtype=float)
        V = np.zeros((mr, rank), dtype=float)
        W = np.zeros((pr, rank), dtype=float)
        
        # Strassen's 7 products for 2x2 matrix multiplication.
        # A_sub is 2x2, B_sub is 2x2, C_sub = A_sub * B_sub.
        # Indices in sub-matrices: (0,0), (0,1), (1,0), (1,1)
        # 
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Define coefficients for U (A-side), V (B-side), W (C-side)
        # Each product r has:
        # U_coeffs: dict {(i,j): coeff}
        # V_coeffs: dict {(j, k_sub): coeff}
        # W_coeffs: dict {(k_sub, i): coeff} -- Note: W index is k*n+i
        
        strassen_products = [
            { # M1
                'U': {(0, 0): 1, (1, 1): 1},
                'V': {(0, 0): 1, (1, 1): 1},
                'W': {(0, 0): 1, (1, 1): 1}
            },
            { # M2
                'U': {(1, 0): 1, (1, 1): 1},
                'V': {(0, 0): 1},
                'W': {(0, 1): 1, (1, 1): -1} # C10 += M2, C11 -= M2? 
                # Let's re-verify C formulas.
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # 
                # M2 contributes to C10 (+1) and C11 (-1).
                # W index: (k_sub, i).
                # C10 is i=1, k_sub=0. W[0*2+1] = W[1].
                # C11 is i=1, k_sub=1. W[1*2+1] = W[3].
                # So W should have { (0,1): 1, (1,1): -1 }.
                # My previous code had: 'C': {(1, 0): 1, (1, 1): -1} which maps to (i=1, k=0) and (i=1, k=1).
                # Wait, the previous code used 'C' dict with keys (i, k_sub)?
                # Let's check the previous code's indexing for W.
                # Previous: `W[(k1 + k_sub) * n + i, t] = coeff`
                # Key in dict was `(i, k_sub)`?
                # Previous code: `for (i, k_sub), coeff in prod['C'].items():`
                # So key is (i, k_sub).
                # For M2: C10 (i=1, k=0) coeff 1. C11 (i=1, k=1) coeff -1.
                # So dict should be {(1, 0): 1, (1, 1): -1}.
                # My list above has {(0, 1): 1, (1, 1): -1}. This is (i=0, k=1) and (i=1, k=1). This is WRONG for M2.
                # Let's stick to the previous code's correct logic.
            },
        ]
        
        # Re-defining carefully based on standard Strassen and previous working code.
        # Previous code:
        # 1. M1: A:{(0,0):1,(1,1):1}, B:{(0,0):1,(1,1):1}, C:{(0,0):1,(1,1):1}
        #    C00+=M1, C11+=M1. Correct.
        # 2. M2: A:{(1,0):1,(1,1):1}, B:{(0,0):1}, C:{(1,0):1,(1,1):-1}
        #    C10+=M2, C11-=M2. Correct.
        # 3. M3: A:{(0,0):1}, B:{(0,1):1,(1,1):-1}, C:{(0,1):1,(1,1):1}
        #    C01+=M3, C11+=M3. Correct.
        # 4. M4: A:{(1,1):1}, B:{(1,0):1,(0,0):-1}, C:{(0,0):1,(1,0):1}
        #    C00+=M4, C10+=M4. Correct.
        # 5. M5: A:{(0,0):1,(0,1):1}, B:{(1,1):1}, C:{(0,0):-1,(0,1):1}
        #    C00-=M5, C01+=M5. Correct.
        # 6. M6: A:{(1,0):1,(0,0):-1}, B:{(0,0):1,(0,1):1}, C:{(1,1):1}
        #    C11+=M6. Correct.
        # 7. M7: A:{(0,1):1,(1,1):-1}, B:{(1,0):1,(1,1):1}, C:{(0,0):1}
        #    C00+=M7. Correct.

        strassen_products = [
            {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1, (1, 1): 1}},
            {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1}, 'C': {(1, 0): 1, (1, 1): -1}},
            {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1}, 'C': {(0, 1): 1, (1, 1): 1}},
            {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1}, 'C': {(0, 0): 1, (1, 0): 1}},
            {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1}, 'C': {(0, 0): -1, (0, 1): 1}},
            {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1}, 'C': {(1, 1): 1}},
            {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1}},
        ]

        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    V[j * p + (k1 + k_sub), t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    W[(k1 + k_sub) * n + i, t] = coeff
        return U, V, W

    # Get the rank-28 solution
    U, V, W = get_strassen_28()
    
    # Verify
    ok, err = verify(U, V, W)
    if not ok:
        # Fallback to schoolbook rank 32 if something went wrong (should not happen)
        U = np.zeros((nr, 32)); V = np.zeros((mr, 32)); W = np.zeros((pr, 32))
        for idx in range(32):
            i = idx // 16
            j = (idx // 8) % 2
            k = idx % 8
            U[i * m + j, idx] = 1.0
            V[j * p + k, idx] = 1.0
            W[k * n + i, idx] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
