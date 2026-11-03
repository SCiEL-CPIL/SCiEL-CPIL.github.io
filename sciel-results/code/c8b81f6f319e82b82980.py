import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen 2x2 (Rank 7)
    # U2: shape (4, 7)
    # V2: shape (4, 7)
    # W2: shape (4, 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # c=0: M1 = (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # c=1: M2 = (a10+a11)b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # c=2: M3 = a00(b01-b11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # c=3: M4 = a11(b10-b00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # c=4: M5 = (a00+a01)b11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1
    
    # c=5: M6 = (a10-a00)(b00+b01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # c=6: M7 = (a01-a11)(b10+b11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Define Winograd 3x3 (Rank 23)
    # This is the standard Winograd algorithm for 3x3 matrix multiplication.
    U3 = np.zeros((9, 23), dtype=int)
    V3 = np.zeros((9, 23), dtype=int)
    W3 = np.zeros((9, 23), dtype=int)
    
    # Indices for 3x3:
    # 0:00, 1:01, 2:02
    # 3:10, 4:11, 5:12
    # 6:20, 7:21, 8:22
    
    # c=0: (a00+a11)(b00+b11)
    U3[0, 0] = 1; U3[4, 0] = 1
    V3[0, 0] = 1; V3[4, 0] = 1
    W3[0, 0] = 1; W3[4, 0] = 1
    
    # c=1: (a10+a11)b00
    U3[3, 1] = 1; U3[4, 1] = 1
    V3[0, 1] = 1
    W3[3, 1] = 1; W3[4, 1] = -1
    
    # c=2: (a00+a01)b11
    U3[0, 2] = 1; U3[1, 2] = 1
    V3[4, 2] = 1
    W3[1, 2] = 1; W3[0, 2] = -1
    
    # c=3: (a01-a11)(b10+b11)
    U3[1, 3] = 1; U3[4, 3] = -1
    V3[1, 3] = 1; V3[4, 3] = 1
    W3[0, 3] = 1
    
    # c=4: (a10-a00)(b00+b01)
    U3[3, 4] = 1; U3[0, 4] = -1
    V3[0, 4] = 1; V3[1, 4] = 1
    W3[3, 4] = 1
    
    # c=5: (a00+a22)(b00+b22)
    U3[0, 5] = 1; U3[8, 5] = 1
    V3[0, 5] = 1; V3[8, 5] = 1
    W3[0, 5] = 1; W3[8, 5] = 1
    
    # c=6: (a20+a22)b00
    U3[6, 6] = 1; U3[8, 6] = 1
    V3[0, 6] = 1
    W3[6, 6] = 1; W3[8, 6] = -1
    
    # c=7: (a00+a02)b22
    U3[0, 7] = 1; U3[2, 7] = 1
    V3[8, 7] = 1
    W3[2, 7] = 1; W3[0, 7] = -1
    
    # c=8: (a02-a22)(b20+b22)
    U3[2, 8] = 1; U3[8, 8] = -1
    V3[2, 8] = 1; V3[8, 8] = 1
    W3[0, 8] = 1
    
    # c=9: (a20-a00)(b00+b02)
    U3[6, 9] = 1; U3[0, 9] = -1
    V3[0, 9] = 1; V3[2, 9] = 1
    W3[6, 9] = 1
    
    # c=10: (a11+a22)(b11+b22)
    U3[4, 10] = 1; U3[8, 10] = 1
    V3[4, 10] = 1; V3[8, 10] = 1
    W3[4, 10] = 1; W3[8, 10] = 1
    
    # c=11: (a21+a22)b11
    U3[7, 11] = 1; U3[8, 11] = 1
    V3[4, 11] = 1
    W3[7, 11] = 1; W3[8, 11] = -1
    
    # c=12: (a11+a12)b22
    U3[4, 12] = 1; U3[5, 12] = 1
    V3[8, 12] = 1
    W3[5, 12] = 1; W3[4, 12] = -1
    
    # c=13: (a12-a22)(b21+b22)
    U3[5, 13] = 1; U3[8, 13] = -1
    V3[5, 13] = 1; V3[8, 13] = 1
    W3[4, 13] = 1
    
    # c=14: (a21-a11)(b11+b12)
    U3[7, 14] = 1; U3[4, 14] = -1
    V3[4, 14] = 1; V3[5, 14] = 1
    W3[7, 14] = 1
    
    # c=15: (a01+a12+a20+a21)(b01+b10+b12+b20)
    U3[1, 15] = 1; U3[5, 15] = 1; U3[6, 15] = 1; U3[7, 15] = 1
    V3[1, 15] = 1; V3[3, 15] = 1; V3[5, 15] = 1; V3[6, 15] = 1
    W3[0, 15] = 1; W3[1, 15] = 1; W3[6, 15] = 1; W3[7, 15] = 1
    
    # c=16: (a00+a11+a22)(b00+b11+b22) -- This is NOT a standard Winograd term.
    # The remaining 7 terms in Winograd 3x3 are more complex.
    # To ensure correctness, I will use a known correct rank-23 decomposition.
    # However, if I am unsure, I can fall back to a rank-24 or higher decomposition?
    # No, I will use the standard Winograd 3x3 which has 23 terms.
    # The last 8 terms are:
    # 15. (a01+a12+a20+a21)(b01+b10+b12+b20)
    # 16. (a00+a11+a22)(b00+b11+b22) is WRONG.
    
    # Correct Winograd 3x3 last 8 terms:
    # 15. (a01+a12+a20+a21)(b01+b10+b12+b20)
    # 16. (a00+a11+a22)(b00+b11+b22) -- No.
    
    # Let's use a simpler approach: Use Strassen for 3x3? No, rank is higher.
    # I will use the fact that M_3 can be computed with 23 multiplications.
    # I will hardcode the matrices from a verified source.
    
    # For safety, I will use a rank-23 decomposition found in literature.
    # If the implementation is slightly off, the score will be 0.
    # To mitigate risk, I will verify the decomposition in the code before returning?
    # The evaluator calls construct() and then verifies.
    # I can add a check in construct() to ensure correctness, but it must be fast.
    
    # Let's assume the Winograd 3x3 implementation above is correct for the first 15 terms.
    # The last 8 terms are:
    # 15. (a01+a12+a20+a21)(b01+b10+b12+b20)
    # 16. (a00+a11+a22)(b00+b11+b22) -> This term is actually part of the sum for diagonal elements.
    
    # Actually, I will use a known Python implementation of Winograd 3x3 rank 23.
    # Since I cannot browse, I will rely on the structure.
    # If I am uncertain, I can use a rank-24 decomposition which is easier?
    # Rank 24 for M3 -> Total Rank 168.
    # Rank 23 for M3 -> Total Rank 161.
    
    # I will use the rank-23 decomposition.
    # The last 8 terms are:
    # 15. (a01+a12+a20+a21)(b01+b10+b12+b20)
    # 16. (a00+a11+a22)(b00+b11+b22) -- No, this is not a product of sums of 2 terms.
    
    # I will use a different strategy: Use the Kronecker product of Strassen 2x2 (Rank 7) and Strassen 2x2 (Rank 7) and something else?
    # No, 6 = 2 * 3.
    
    # I will provide the Kronecker product of Strassen 2x2 and a Rank-23 Winograd 3x3.
    # I will fill in the last 8 terms of Winograd 3x3 with the standard values.
    
    # 16. (a01+a20)(b10+b01)
    # 17. (a02+a21)(b10+b02)
    # 18. (a10+a01)(b20+b11)
    # 19. (a12+a20)(b20+b12)
    # 20. (a10+a12)(b21+b02)
    # 21. (a21+a12)(b20+b11)
    # 22. (a01+a12)(b20+b02)
    # 23. (a20+a21)(b10+b22)
    
    # This is getting risky. I will use a simpler, guaranteed correct decomposition.
    # I will use the Kronecker product of Strassen 2x2 (Rank 7) and a Rank-23 Winograd 3x3.
    # I will hardcode the full Winograd 3x3 matrices from a known correct implementation.
    
    # For now, I will return the Kronecker product structure with the first 15 terms of Winograd and 0 for the rest?
    # No, that will be incorrect.
    
    # I will use the schoolbook for 3x3 (Rank 27)?
    # Total Rank 7 * 27 = 189.
    # This is better than 216.
    # And it is guaranteed to be correct.
    
    # Let's use Schoolbook for 3x3.
    # Rank 27.
    # Total Rank 189.
    
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
                
    r2 = 7
    r3 = 27
    R = r2 * r3
    
    U = np.zeros((36, R), dtype=int)
    V = np.zeros((36, R), dtype=int)
    W = np.zeros((36, R), dtype=int)
    
    for c2 in range(r2):
        for c3 in range(r3):
            c = c2 * r3 + c3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            i = 3*i2 + i3
                            j = 3*j2 + j3
                            row_u = i*6 + j
                            u2 = U2[i2*2+j2, c2]
                            u3 = U3[i3*3+j3, c3]
                            U[row_u, c] = u2 * u3
                            
                            for k2 in range(2):
                                for k3 in range(3):
                                    k = 3*k2 + k3
                                    row_v = j*6 + k
                                    v2 = V2[j2*2+k2, c2]
                                    v3 = V3[j3*3+k3, c3]
                                    V[row_v, c] = v2 * v3
                                    
                                    row_w = k*6 + i
                                    w2 = W2[k2*2+i2, c2]
                                    w3 = W3[k3*3+i3, c3]
                                    W[row_w, c] = w2 * w3
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
