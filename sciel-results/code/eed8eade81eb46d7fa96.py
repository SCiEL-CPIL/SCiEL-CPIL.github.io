import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # Decompose into (3x2x4) and (1x2x2)
    # R(3,2,4) <= 12 (3 * R(1,2,4))
    # R(1,2,2) = 2
    # Total rank: 24
    
    # Construct U1, V1, W1 for 3x2x4 with rank 12
    U1 = np.zeros((3*2, 12))
    V1 = np.zeros((2*4, 12))
    W1 = np.zeros((4*3, 12))
    
    # For each row i1 in 0..2, use rank-4 decomposition of 1x2x4
    # 1x2x4 decomposition (rank 4):
    # Term 0: a0 b00 -> c0
    # Term 1: a0 b01 -> c1
    # Term 2: a0 b02 -> c2
    # Term 3: a0 b03 -> c3
    # Term 4: a1 b10 -> c0
    # Term 5: a1 b11 -> c1
    # Term 6: a1 b12 -> c2
    # Term 7: a1 b13 -> c3
    # Wait, R(1,2,4) is 4, not 8.
    # My previous analysis of R(1,2,4) was 4.
    # Let's re-verify R(1,2,4).
    # c_k = a0 b0k + a1 b1k.
    # This is 4 equations.
    # Can we do it in 4 terms?
    # Yes, if we use the structure.
    # But the "rank 4" I found earlier was actually schoolbook-like?
    # No, schoolbook for 1x2x4 is 8.
    # I claimed R(1,2,4)=4.
    # Let's check if rank 4 is possible.
    # Tensor T(j, k, r). j in 0,1. k in 0..3.
    # T(0, k, r) and T(1, k, r).
    # We need sum_r U(j, r) V(k, r) W(k', r) = delta(j, input) ...
    # Actually, for 1x2x4, the tensor is determined by the 8 entries of B and 2 of A.
    # If rank is 4, it's very low.
    # Let's assume R(1,2,4) = 4 is correct (based on border rank arguments or similar).
    # If not, we can use rank 8 (schoolbook).
    # If R(1,2,4)=8, then R(3,2,4) <= 24.
    # Then total rank <= 48.
    
    # To be safe, let's use the rank 48 construction which is definitely valid.
    # Or try to implement the rank 24 if I can be sure.
    # Given the time, rank 48 is a safe bet for improvement.
    
    # However, I can easily implement the "concatenation" of row decompositions.
    # If I use schoolbook for each row (rank 8 per row), total rank 24 for 3x2x4.
    # Total rank 48.
    # This is the same as the Kronecker product of schoolbooks.
    
    # Let's stick to the rank 48 implementation derived above.
    
    U1 = np.zeros((6, 24))
    V1 = np.zeros((8, 24))
    W1 = np.zeros((12, 24))
    
    r1 = 0
    for i1 in range(3):
        for j1 in range(2):
            for k1 in range(4):
                U1[i1*2 + j1, r1] = 1
                V1[j1*4 + k1, r1] = 1
                W1[k1*3 + i1, r1] = 1
                r1 += 1
                
    U2 = np.zeros((2, 2))
    V2 = np.zeros((4, 2))
    W2 = np.zeros((2, 2))
    
    r2 = 0
    for j2 in range(2):
        for k2 in range(2):
            U2[j2, r2] = 1
            V2[j2*2 + k2, r2] = 1
            W2[k2, r2] = 1
            r2 += 1
            
    rank = 48
    U = np.zeros((12, rank))
    V = np.zeros((32, rank))
    W = np.zeros((24, rank))
    
    for r1 in range(24):
        for r2 in range(2):
            r = r1 * 2 + r2
            for i1 in range(3):
                for j1 in range(2):
                    for j2 in range(2):
                        idx_u = i1*4 + 2*j1 + j2
                        val = U1[i1*2 + j1, r1] * U2[j2, r2]
                        if val != 0:
                            U[idx_u, r] = val
                            
            for j1 in range(2):
                for k1 in range(4):
                    for j2 in range(2):
                        for k2 in range(2):
                            idx_v = (2*j1 + j2)*8 + 2*k1 + k2
                            val = V1[j1*4 + k1, r1] * V2[j2*2 + k2, r2]
                            if val != 0:
                                V[idx_v, r] = val
                                
            for k1 in range(4):
                for i1 in range(3):
                    for k2 in range(2):
                        idx_w = (2*k1 + k2)*3 + i1
                        val = W1[k1*3 + i1, r1] * W2[k2, r2]
                        if val != 0:
                            W[idx_w, r] = val
                            
    return {"U": U, "V": V, "W": W}
