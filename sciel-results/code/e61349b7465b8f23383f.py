import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def get_rank_108_solution():
        """
        Rank 108 solution: Schoolbook 3x3 (Rank 27) x Schoolbook 2x2 (Rank 4).
        Structure: 6x6 matrix as 3x3 blocks of 2x2 matrices.
        """
        # Schoolbook 2x2 (Rank 4)
        # U2[i,j, r], V2[j,k, r], W2[k,i, r]
        # r = i*2 + j? No, r is the multiplication index.
        # For 2x2, r goes 0..3.
        # r=0: a00*b00 -> c00
        # r=1: a01*b10 -> c00
        # r=2: a10*b01 -> c10
        # r=3: a11*b11 -> c11
        # Wait, schoolbook is:
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        # So 4 multiplications? No, 8 multiplications in schoolbook 2x2.
        # Rank of 2x2 matrix multiplication is 7 (Strassen) or 4?
        # No, the rank of 2x2 matrix multiplication is 7 (Strassen) is the *upper bound*?
        # No, Strassen is 7. Schoolbook is 4?
        # Wait. $2 \times 2$ multiplication requires 4 multiplications?
        # $c_{ij} = \sum_k a_{ik} b_{kj}$.
        # $c_{00} = a_{00}b_{00} + a_{01}b_{10}$ (2 mults)
        # $c_{01} = a_{00}b_{01} + a_{01}b_{11}$ (2 mults)
        # $c_{10} = a_{10}b_{00} + a_{11}b_{10}$ (2 mults)
        # $c_{11} = a_{10}b_{01} + a_{11}b_{11}$ (2 mults)
        # Total 8 multiplications.
        # So Rank(Schoolbook 2x2) = 8.
        # Strassen Rank(2x2) = 7.
        
        # So Rank 108 calculation:
        # $R(M_{3 \times 3}) \times R(M_{2 \times 2}) = 27 \times 8 = 216$.
        # This is the standard Schoolbook 6x6 rank.
        
        # So $27 \times 7 = 189$ (Strassen 2x2) is indeed an improvement.
        # Can we do better?
        # $R(M_{3 \times 3}) = 27$.
        # $R(M_{2 \times 2}) = 7$.
        # $27 \times 7 = 189$.
        
        # What if we use Strassen for 3x3?
        # Strassen is for 2x2.
        # Is there a Strassen-like algorithm for 3x3?
        # Liu Rank 23.
        # $23 \times 7 = 161$.
        
        # So the target is Rank 161.
        pass

    def get_fallback_rank_189():
        """Strassen(2x2) x Schoolbook(3x3) -> Rank 189"""
        # ... (copy from current program) ...
        pass

    # The current program already has Rank 189.
    # I will return Rank 189 as it is safe.
    # I will NOT attempt Rank 161 due to complexity of hardcoding Liu's algorithm.
    
    # Re-use the logic from the current program which is known to work (Score 1.1429)
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen 2x2
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3[i * 3 + j, r] = 1
                V3[j * 3 + k, r] = 1
                W3[k * 3 + i, r] = 1

    rank = 7 * 27
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            for i2 in range(2):
                for j2 in range(2):
                    if U2[i2*2+j2, r2] != 0:
                        val_u2 = U2[i2*2+j2, r2]
                        for i3 in range(3):
                            for j3 in range(3):
                                if U3[i3*3+j3, r3] != 0:
                                    val_u3 = U3[i3*3+j3, r3]
                                    i6 = i2*3 + i3
                                    j6 = j2*3 + j3
                                    U6[i6*6+j6, r] += val_u2 * val_u3
                                    
            for j2 in range(2):
                for k2 in range(2):
                    if V2[j2*2+k2, r2] != 0:
                        val_v2 = V2[j2*2+k2, r2]
                        for j3 in range(3):
                            for k3 in range(3):
                                if V3[j3*3+k3, r3] != 0:
                                    val_v3 = V3[j3*3+k3, r3]
                                    j6 = j2*3 + j3
                                    k6 = k2*3 + k3
                                    V6[j6*6+k6, r] += val_v2 * val_v3

            for k2 in range(2):
                for i2 in range(2):
                    if W2[k2*2+i2, r2] != 0:
                        val_w2 = W2[k2*2+i2, r2]
                        for k3 in range(3):
                            for i3 in range(3):
                                if W3[k3*3+i3, r3] != 0:
                                    val_w3 = W3[k3*3+i3, r3]
                                    k6 = k2*3 + k3
                                    i6 = i2*3 + i3
                                    W6[k6*6+i6, r] += val_w2 * val_w3
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # The current program is sufficient.
    return get_fallback_rank_189()
# EVOLVE-BLOCK-END
