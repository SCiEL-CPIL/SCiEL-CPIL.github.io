import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Winograd Fourier Matrix Multiplication (FMM)
    # Rank = N^2 = 36
    N = 6
    
    # DFT matrix F_N
    # F[u, k] = exp(-2j * pi * u * k / N)
    # We use real arithmetic by separating real and imaginary parts?
    # Or just use complex? The task requires integer/half-integer entries.
    # DFT entries are not integers.
    # Winograd FMM usually uses integer coefficients for specific N.
    # For N=6, the DFT involves sqrt(3) and i.
    # This violates the "integer/half-integer entries" constraint if we output the raw DFT.
    # However, the tensor decomposition can be scaled.
    # But the problem states "integer/half-integer entries".
    # Winograd FMM for N=6 might not have integer coefficients.
    
    # Let's check N=4. Rank 16.
    # N=2. Rank 4.
    # N=6. Rank 36.
    
    # If Winograd FMM coefficients are not integers, we cannot use it directly.
    # We must stick to Strassen (Rank 7) and Schoolbook (Rank 27).
    # Rank 189.
    
    # Is there a better decomposition?
    # Maybe 3x3 Strassen-like?
    # Rank of 3x3 matrix multiplication is at most 23 (Winograd).
    # 23 * 7 = 161.
    # Winograd 3x3 decomposition has rank 23.
    # Coefficients are integers?
    # Winograd 3x3:
    # Uses 23 multiplications.
    # Coefficients are 0, 1, -1.
    # This is valid.
    
    # Let's use Winograd 3x3 (Rank 23) and Strassen 2x2 (Rank 7).
    # Total Rank = 23 * 7 = 161.
    # Improvement over 189.
    
    # Winograd 3x3 factors:
    # U3: 9x23, V3: 9x23, W3: 9x23
    # Strassen 2x2 factors:
    # U2: 4x7, V2: 4x7, W2: 4x7
    
    # Construct 6x6 factors.
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Winograd 3x3 Factors
    # Reference: Winograd, S. (1978). "On the complexity of a family of matrix multiplication algorithms"
    # Rank 23.
    # U3, V3, W3 are 9x23.
    # Indices: i,j for U (0..8), j,k for V (0..8), k,i for W (0..8).
    
    U3 = np.zeros((9, 23), dtype=int)
    V3 = np.zeros((9, 23), dtype=int)
    W3 = np.zeros((9, 23), dtype=int)
    
    # Winograd 3x3 decomposition details
    # This is complex to hardcode correctly from memory.
    # I will use the known 23-multiplication algorithm.
    
    # If I cannot recall the exact 23 factors, I will stick to Schoolbook 3x3 (Rank 27).
    # Rank 189 is safe.
    # But 161 is better.
    
    # Let's try to implement Winograd 3x3.
    # Factors are available in literature.
    # For now, I will use Schoolbook 3x3 to ensure correctness, as Winograd 3x3 is error-prone to type.
    # Score 1.14 is from Rank 189.
    # If I can get Rank 161, score is 1.14 * 189/161 = 1.34.
    
    # I will stick to the current Rank 189 implementation but optimize the code for speed.
    # The current code is already quite fast.
    
    # Is there a Rank < 189?
    # 6x6 can be decomposed as 2x2 blocks.
    # Rank 189.
    
    # I will return the current best known Rank 189.
    
    rank = 7 * 27
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            
            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            
            # Schoolbook 3x3
            # r3 = i3 * 9 + j3 * 3 + k3
            i3 = r3 // 9
            j3 = (r3 % 9) // 3
            k3 = r3 % 3
            
            # U3[i3*3+j3, r3] = 1
            # V3[j3*3+k3, r3] = 1
            # W3[k3*3+i3, r3] = 1
            
            for u2 in nz_u2:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                i6 = i2 * 3 + i3
                j6 = j2 * 3 + j3
                g_u = i6 * 6 + j6
                U6[g_u, r] = val_u2

            for v2 in nz_v2:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                j6 = j2 * 3 + j3
                k6 = k2 * 3 + k3
                g_v = j6 * 6 + k6
                V6[g_v, r] = val_v2

            for w2 in nz_w2:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                k6 = k2 * 3 + k3
                i6 = i2 * 3 + i3
                g_w = k6 * 6 + i6
                W6[g_w, r] = val_w2

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
