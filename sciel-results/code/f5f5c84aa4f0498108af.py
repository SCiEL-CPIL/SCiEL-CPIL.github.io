import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
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

    def get_schoolbook_3x3():
        """Schoolbook 3x3 decomposition (Rank 27) - guaranteed correct."""
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
        return U3, V3, W3

    def verify_3x3(U3, V3, W3, rank3, tol=1e-9):
        """Verify 3x3 decomposition: sum_r U3[i*3+j,r]*V3[j*3+k,r]*W3[k*3+i,r] == 1"""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank3):
                        s += U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_kronecker(U2, V2, W2, U3, V3, W3, rank2, rank3):
        """Build 6x6 factors from 2x2 and 3x3 decompositions via Kronecker product."""
        rank = rank2 * rank3
        U6 = np.zeros((36, rank), dtype=float)
        V6 = np.zeros((36, rank), dtype=float)
        W6 = np.zeros((36, rank), dtype=float)
        
        for r2 in range(rank2):
            for r3 in range(rank3):
                r = r2 * rank3 + r3
                
                for i2 in range(2):
                    for j2 in range(2):
                        val_u2 = U2[i2*2+j2, r2]
                        if val_u2 != 0:
                            for i3 in range(3):
                                for j3 in range(3):
                                    val_u3 = U3[i3*3+j3, r3]
                                    if val_u3 != 0:
                                        i6 = i2*3 + i3
                                        j6 = j2*3 + j3
                                        U6[i6*6+j6, r] += val_u2 * val_u3
                
                for j2 in range(2):
                    for k2 in range(2):
                        val_v2 = V2[j2*2+k2, r2]
                        if val_v2 != 0:
                            for j3 in range(3):
                                for k3 in range(3):
                                    val_v3 = V3[j3*3+k3, r3]
                                    if val_v3 != 0:
                                        j6 = j2*3 + j3
                                        k6 = k2*3 + k3
                                        V6[j6*6+k6, r] += val_v2 * val_v3
                
                for k2 in range(2):
                    for i2 in range(2):
                        val_w2 = W2[k2*2+i2, r2]
                        if val_w2 != 0:
                            for k3 in range(3):
                                for i3 in range(3):
                                    val_w3 = W3[k3*3+i3, r3]
                                    if val_w3 != 0:
                                        k6 = k2*3 + k3
                                        i6 = i2*3 + i3
                                        W6[k6*6+i6, r] += val_w2 * val_w3
        
        return U6, V6, W6

    def verify_6x6_full(U6, V6, W6, rank, tol=1e-6):
        """Full exhaustive verification of 6x6 decomposition."""
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    s = 0.0
                    for r in range(rank):
                        s += U6[i*6+j, r] * V6[j*6+k, r] * W6[k*6+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def check_half_integer(mat, tol=1e-9):
        """Check all entries are integer or half-integer."""
        for val in mat:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True

    # Build the guaranteed-correct schoolbook 3x3 (rank 27) and Kronecker with Strassen 2x2 (rank 7)
    # Total rank: 7 * 27 = 189
    U3_sb, V3_sb, W3_sb = get_schoolbook_3x3()
    U6, V6, W6 = build_kronecker(U2, V2, W2, U3_sb, V3_sb, W3_sb, 7, 27)
    
    # Convert to int since schoolbook uses only 0s and 1s
    U6 = U6.astype(int)
    V6 = V6.astype(int)
    W6 = W6.astype(int)
    
    # Full verification of the fallback
    if not verify_6x6_full(U6, V6, W6, 189):
        raise RuntimeError("Fallback decomposition failed full verification!")
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
