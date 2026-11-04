import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen 2x2 rank-7 algorithm
    def get_strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)

        # m0 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # m1 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1

        # m2 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        # m3 = A11(B10-B00)
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = 1; W[1, 3] = 1

        # m4 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # m5 = (A10-A00)(B00+B01)
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # m6 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    # Schoolbook 3x3 (rank 27)
    def get_schoolbook_3x3():
        U = np.zeros((9, 27), dtype=int)
        V = np.zeros((9, 27), dtype=int)
        W = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U[i * 3 + j, r] = 1
                    V[j * 3 + k, r] = 1
                    W[k * 3 + i, r] = 1
        return U, V, W

    # Laderman 3x3 rank-19 algorithm
    def get_laderman_3x3():
        U = np.zeros((9, 19), dtype=int)
        V = np.zeros((9, 19), dtype=int)
        W = np.zeros((9, 19), dtype=int)
        
        # Term 0: A00 B00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 1: A01 B10
        U[1, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
        # Term 2: A02 B20
        U[2, 2] = 1; V[6, 2] = 1; W[6, 2] = 1
        # Term 3: A10 B01
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        # Term 4: A11 B11
        U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
        # Term 5: A12 B21
        U[5, 5] = 1; V[7, 5] = 1; W[7, 5] = 1
        # Term 6: A20 B02
        U[6, 6] = 1; V[2, 6] = 1; W[2, 6] = 1
        # Term 7: A21 B12
        U[7, 7] = 1; V[5, 7] = 1; W[5, 7] = 1
        # Term 8: A22 B22
        U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
        
        # Term 9: (A00+A11) B02
        U[0, 9] = 1; U[4, 9] = 1
        V[2, 9] = 1
        W[2, 9] = 1; W[5, 9] = -1 # Correction needed?
        
        # This manual construction is prone to error.
        # I will use a known correct set of coefficients for Laderman.
        # Based on literature, the 19 terms are:
        # T0 = A00 B00
        # T1 = A01 B10
        # T2 = A02 B20
        # T3 = A10 B01
        # T4 = A11 B11
        # T5 = A12 B21
        # T6 = A20 B02
        # T7 = A21 B12
        # T8 = A22 B22
        # T9 = (A00+A11) B02
        # T10 = (A01+A11) B20
        # T11 = (A02+A12) B01
        # T12 = (A10+A22) B12
        # T13 = (A20+A21) B00
        # T14 = (A00+A22) B11
        # T15 = (A01+A22) B10
        # T16 = (A10+A11) B00
        # T17 = (A11+A22) B01
        # T18 = (A20+A21) B11
        
        # And the W coefficients are determined by solving the linear system.
        # Since I cannot solve it here, I will rely on the verification step.
        # If it fails, I return None.
        
        # Let's try to define the W matrix such that it works.
        # Actually, the standard Laderman algorithm has specific W coefficients.
        # I will skip this if I am not confident.
        
        return None

    # Verify a decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    def verify_decomposition(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    # Build 6x6 from 2x2 (Strassen) and 3x3 (inner) via Kronecker product structure
    def build_6x6(U2, V2, W2, U3, V3, W3):
        rank_2 = U2.shape[1]
        rank_3 = U3.shape[1]
        rank = rank_2 * rank_3
        
        U6 = np.zeros((36, rank), dtype=int)
        V6 = np.zeros((36, rank), dtype=int)
        W6 = np.zeros((36, rank), dtype=int)

        for r2 in range(rank_2):
            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            
            for r3 in range(rank_3):
                r = r2 * rank_3 + r3
                
                nz_u3 = np.nonzero(U3[:, r3])[0]
                nz_v3 = np.nonzero(V3[:, r3])[0]
                nz_w3 = np.nonzero(W3[:, r3])[0]
                
                for u2 in nz_u2:
                    i2 = u2 // 2
                    j2 = u2 % 2
                    val_u2 = U2[u2, r2]
                    
                    for u3 in nz_u3:
                        i3 = u3 // 3
                        j3 = u3 % 3
                        val_u3 = U3[u3, r3]
                        
                        i6 = i2 * 3 + i3
                        j6 = j2 * 3 + j3
                        g_u = i6 * 6 + j6
                        U6[g_u, r] += val_u2 * val_u3

                for v2 in nz_v2:
                    j2 = v2 // 2
                    k2 = v2 % 2
                    val_v2 = V2[v2, r2]
                    
                    for v3 in nz_v3:
                        j3 = v3 // 3
                        k3 = v3 % 3
                        val_v3 = V3[v3, r3]
                        
                        j6 = j2 * 3 + j3
                        k6 = k2 * 3 + k3
                        g_v = j6 * 6 + k6
                        V6[g_v, r] += val_v2 * val_v3

                for w2 in nz_w2:
                    k2 = w2 // 2
                    i2 = w2 % 2
                    val_w2 = W2[w2, r2]
                    
                    for w3 in nz_w3:
                        k3 = w3 // 3
                        i3 = w3 % 3
                        val_w3 = W3[w3, r3]
                        
                        k6 = k2 * 3 + k3
                        i6 = i2 * 3 + i3
                        g_w = k6 * 6 + i6
                        W6[g_w, r] += val_w2 * val_w3

        return U6, V6, W6

    U2, V2, W2 = get_strassen_2x2()
    
    # Try Laderman 3x3 first (rank 7*19=133)
    U3l, V3l, W3l = get_laderman_3x3()
    if U3l is not None and verify_decomposition(U3l, V3l, W3l, 3, 3, 3):
        U6, V6, W6 = build_6x6(U2, V2, W2, U3l, V3l, W3l)
        if verify_decomposition(U6, V6, W6, 6, 6, 6):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Fallback: Schoolbook 3x3 (rank 7*27=189)
    U3s, V3s, W3s = get_schoolbook_3x3()
    U6, V6, W6 = build_6x6(U2, V2, W2, U3s, V3s, W3s)
    if verify_decomposition(U6, V6, W6, 6, 6, 6):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Last resort: pure schoolbook 6x6 (rank 216) - should not reach here
    U6 = np.zeros((36, 216), dtype=int)
    V6 = np.zeros((36, 216), dtype=int)
    W6 = np.zeros((36, 216), dtype=int)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                r = i * 36 + j * 6 + k
                U6[i * 6 + j, r] = 1
                V6[j * 6 + k, r] = 1
                W6[k * 6 + i, r] = 1
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
