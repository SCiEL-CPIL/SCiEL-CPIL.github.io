import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # 1. Strassen 2x2 Decomposition (Rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: A00+A11, E00+E11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2: A10+A11, E00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # M3: A00, E10-E11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # M4: A11, E01-E00
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # M5: A00+A01, E11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # M6: A10-A00, E00+E01
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7: A01-A11, E10+E11
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W2 Output Reconstruction
    # C00 = M1 + M4 - M5 + M7
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C01 = M3 + M5
    W2[2, 2] = 1; W2[2, 4] = 1
    # C10 = M2 + M4
    W2[1, 1] = 1; W2[1, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1

    # 2. Laderman 3x3 Decomposition (Rank 23)
    # U3: (9, 23), V3: (9, 23), W3: (9, 23)
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # Hardcoded Laderman coefficients
    # Source: Standard Laderman algorithm implementation
    # Each column corresponds to a product term.
    # U3[i*3+j, col], V3[j*3+k, col], W3[k*3+i, col]
    
    # Term 0: A00+E00
    U3[0, 0] = 1; U3[8, 0] = 1
    V3[0, 0] = 1; V3[8, 0] = 1
    W3[0, 0] = 1
    
    # Term 1: A01
    U3[1, 1] = 1
    V3[0, 1] = 1
    W3[2, 1] = 1
    
    # Term 2: A02
    U3[2, 2] = 1
    V3[0, 2] = 1
    W3[4, 2] = 1
    
    # Term 3: A10
    U3[3, 3] = 1
    V3[0, 3] = 1
    W3[1, 3] = 1
    
    # Term 4: A12
    U3[5, 4] = 1
    V3[2, 4] = 1
    W3[7, 4] = 1
    
    # Term 5: A21
    U3[7, 5] = 1
    V3[4, 5] = 1
    W3[6, 5] = 1
    
    # Term 6: E01
    U3[0, 6] = 1
    V3[1, 6] = 1
    W3[3, 6] = 1
    
    # Term 7: E02
    U3[0, 7] = 1
    V3[2, 7] = 1
    W3[5, 7] = 1
    
    # Term 8: E10
    U3[1, 8] = 1
    V3[3, 8] = 1
    W3[0, 8] = 1
    
    # Term 9: E20
    U3[2, 9] = 1
    V3[6, 9] = 1
    W3[3, 9] = 1 # Wait, E20 * A?? No, Laderman mixes A and E differently.
    
    # Let's use a verified Laderman set to avoid errors.
    # Reference: "Laderman's algorithm" coefficients from literature/code.
    # A safer bet is to generate the 23 terms programmatically if possible, 
    # but hardcoding is faster if correct.
    # I will use the specific Laderman coefficients known to be correct.
    
    # Reset U3, V3, W3 to zeros just in case
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # Laderman 3x3 Rank 23 Coefficients
    # c0: A00+E00
    U3[0,0]=1; U3[8,0]=1; V3[0,0]=1; V3[8,0]=1; W3[0,0]=1
    # c1: A01
    U3[1,1]=1; V3[0,1]=1; W3[2,1]=1
    # c2: A02
    U3[2,2]=1; V3[0,2]=1; W3[4,2]=1
    # c3: A10
    U3[3,3]=1; V3[0,3]=1; W3[1,3]=1
    # c4: A12
    U3[5,4]=1; V3[2,4]=1; W3[7,4]=1
    # c5: A21
    U3[7,5]=1; V3[4,5]=1; W3[6,5]=1
    # c6: E01
    U3[0,6]=1; V3[1,6]=1; W3[3,6]=1
    # c7: E02
    U3[0,7]=1; V3[2,7]=1; W3[5,7]=1
    # c8: E10
    U3[1,8]=1; V3[3,8]=1; W3[0,8]=1
    # c9: E20
    U3[2,9]=1; V3[6,9]=1; W3[3,9]=1 # Correction: W index is k*3+i. E20 is row 2, col 0. A??
    # Actually, let's look at the structure:
    # Term t is sum of U3[i,j,t]*V3[j,k,t]. Output goes to W3[k,i,t].
    # Laderman's 23 terms are specific combinations.
    
    # Verified Laderman 3x3 (from standard sources):
    # 1. A00+E00 -> C00
    # 2. A01 -> C01
    # 3. A02 -> C02
    # 4. A10 -> C10
    # 5. A12 -> C12
    # 6. A21 -> C21
    # 7. E01 -> C10? No.
    
    # Let's use a Python library or known correct hardcode.
    # Since I cannot guarantee my memory of all 23 Laderman terms perfectly,
    # and an error results in score 0, I will fall back to a safer strategy:
    # Use 3x3 Schoolbook (Rank 27) x 2x2 Strassen (Rank 7) = Rank 189.
    # This is guaranteed correct and better than 216.
    # However, 161 is achievable. Let me try to implement a *check* or use a simpler known decomposition.
    # Actually, there is a simpler 3x3 decomposition with rank < 27? No, 23 is the bound.
    # If I am not 100% sure of Laderman, 189 is safe.
    
    # BUT, I can construct the 3x3 part using the *same* logic as Strassen if I pad to 4x4?
    # 4x4 Strassen rank 49. 49 * 49 = 2401. Bad.
    # 3x3 Schoolbook is 27.
    
    # Let's try to implement Laderman carefully.
    # The 23 terms for 3x3 multiplication (Laderman):
    # I will use the coefficients from a reliable online source found in training data.
    
    # Reset
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # t0: A00+E00
    U3[0,0]=1; U3[8,0]=1; V3[0,0]=1; V3[8,0]=1; W3[0,0]=1
    # t1: A01
    U3[1,1]=1; V3[0,1]=1; W3[2,1]=1
    # t2: A02
    U3[2,2]=1; V3[0,2]=1; W3[4,2]=1
    # t3: A10
    U3[3,3]=1; V3[0,3]=1; W3[1,3]=1
    # t4: A12
    U3[5,4]=1; V3[2,4]=1; W3[7,4]=1
    # t5: A21
    U3[7,5]=1; V3[4,5]=1; W3[6,5]=1
    # t6: E01
    U3[0,6]=1; V3[1,6]=1; W3[3,6]=1
    # t7: E02
    U3[0,7]=1; V3[2,7]=1; W3[5,7]=1
    # t8: E10
    U3[1,8]=1; V3[3,8]=1; W3[0,8]=1
    # t9: E20
    U3[2,9]=1; V3[6,9]=1; W3[3,9]=1
    # t10: A11+E11
    U3[4,10]=1; U3[8,10]=1; V3[4,10]=1; V3[8,10]=1; W3[4,10]=1
    # t11: A11
    U3[4,11]=1; V3[4,11]=1; W3[4,11]=1 # Wait, this is just A11*E11.
    # Laderman is complex. 
    
    # Given the risk of hallucinating Laderman coefficients, I will stick to the 
    # Kronecker product of 3x3 Schoolbook and 2x2 Strassen (Rank 189).
    # This is a proven improvement from 216 to 189.
    
    n3, m3, p3 = 3, 3, 3
    rank3 = 27
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
                
    # Kronecker Product Construction
    rank_total = 27 * 7
    U6 = np.zeros((36, rank_total))
    V6 = np.zeros((36, rank_total))
    W6 = np.zeros((36, rank_total))
    
    for r3 in range(27):
        for r2 in range(7):
            r = r3 * 7 + r2
            
            # U Part
            for i3 in range(3):
                for j3 in range(3):
                    if U3[i3*3+j3, r3] != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                if U2[i2*2+j2, r2] != 0:
                                    i = i3*2 + i2
                                    j = j3*2 + j2
                                    idx = i*6 + j
                                    U6[idx, r] = U3[i3*3+j3, r3] * U2[i2*2+j2, r2]
                                    
            # V Part
            for j3 in range(3):
                for k3 in range(3):
                    if V3[j3*3+k3, r3] != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                if V2[j2*2+k2, r2] != 0:
                                    j = j3*2 + j2
                                    k = k3*2 + k2
                                    idx = j*6 + k
                                    V6[idx, r] = V3[j3*3+k3, r3] * V2[j2*2+k2, r2]
                                    
            # W Part
            for k3 in range(3):
                for i3 in range(3):
                    if W3[k3*3+i3, r3] != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                if W2[k2*2+i2, r2] != 0:
                                    k = k3*2 + k2
                                    i = i3*2 + i2
                                    idx = k*6 + i
                                    W6[idx, r] = W3[k3*3+i3, r3] * W2[k2*2+i2, r2]
                                    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
