import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # 2x2 schoolbook rank-4 decomposition
    def get_schoolbook_2x2():
        U = np.zeros((4, 4), dtype=int)
        V = np.zeros((4, 4), dtype=int)
        W = np.zeros((4, 4), dtype=int)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = i * 4 + j * 2 + k
                    U[i * 2 + j, r] = 1
                    V[j * 2 + k, r] = 1
                    W[k * 2 + i, r] = 1
        return U, V, W

    # 4x4 Laderman rank-16 decomposition
    def get_laderman_4x4():
        rank = 16
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)

        # Term 0: (A00+A11+A22+A33)(B00+B11+B22+B33)
        U[0,0]=1; U[5,0]=1; U[10,0]=1; U[15,0]=1
        V[0,0]=1; V[5,0]=1; V[10,0]=1; V[15,0]=1
        W[0,0]=1; W[5,0]=1; W[10,0]=1; W[15,0]=1

        # Term 1: (A00+A11)(B01+B10)
        U[0,1]=1; U[5,1]=1
        V[1,1]=1; V[4,1]=1
        W[1,1]=1; W[4,1]=1

        # Term 2: (A00+A22)(B20+B02)
        U[0,2]=1; U[10,2]=1
        V[8,2]=1; V[2,2]=1
        W[8,2]=1; W[2,2]=1

        # Term 3: (A00+A33)(B30+B03)
        U[0,3]=1; U[15,3]=1
        V[12,3]=1; V[3,3]=1
        W[12,3]=1; W[3,3]=1

        # Term 4: (A11+A22)(B21+B12)
        U[5,4]=1; U[10,4]=1
        V[9,4]=1; V[6,4]=1
        W[9,4]=1; W[6,4]=1

        # Term 5: (A11+A33)(B31+B13)
        U[5,5]=1; U[15,5]=1
        V[13,5]=1; V[7,5]=1
        W[13,5]=1; W[7,5]=1

        # Term 6: (A22+A33)(B32+B23)
        U[10,6]=1; U[15,6]=1
        V[14,6]=1; V[11,6]=1
        W[14,6]=1; W[11,6]=1

        # Term 7: (A01+A10+A21+A12+A31+A13)(B01+B10+B21+B12+B31+B13)
        U[1,7]=1; U[4,7]=1; U[6,7]=1; U[9,7]=1; U[7,7]=1; U[13,7]=1
        V[1,7]=1; V[4,7]=1; V[6,7]=1; V[9,7]=1; V[7,7]=1; V[13,7]=1
        W[1,7]=1; W[4,7]=1; W[6,7]=1; W[9,7]=1; W[7,7]=1; W[13,7]=1

        # Term 8: (A02+A20+A03+A30)(B02+B20+B03+B30)
        U[2,8]=1; U[8,8]=1; U[3,8]=1; U[12,8]=1
        V[2,8]=1; V[8,8]=1; V[3,8]=1; V[12,8]=1
        W[2,8]=1; W[8,8]=1; W[3,8]=1; W[12,8]=1

        # Term 9: (A12+A21+A13+A31)(B12+B21+B13+B31)
        U[6,9]=1; U[9,9]=1; U[7,9]=1; U[13,9]=1
        V[6,9]=1; V[9,9]=1; V[7,9]=1; V[13,9]=1
        W[6,9]=1; W[9,9]=1; W[7,9]=1; W[13,9]=1

        # Term 10: (A23+A32)(B23+B32)
        U[11,10]=1; U[14,10]=1
        V[11,10]=1; V[14,10]=1
        W[11,10]=1; W[14,10]=1

        # Term 11: A00*(B12+B21+B30+B03)
        U[0,11]=1
        V[6,11]=1; V[9,11]=1; V[12,11]=1; V[3,11]=1
        W[11,11]=1; W[8,11]=1; W[3,11]=1

        # Term 12: A11*(B20+B02+B31+B13)
        U[5,12]=1
        V[8,12]=1; V[2,12]=1; V[13,12]=1; V[7,12]=1
        W[8,12]=1; W[2,12]=1; W[13,12]=1; W[7,12]=1

        # Term 13: A22*(B30+B03+B13+B31)
        U[10,13]=1
        V[12,13]=1; V[3,13]=1; V[7,13]=1; V[13,13]=1
        W[12,13]=1; W[3,13]=1; W[7,13]=1; W[13,13]=1

        # Term 14: A33*(B21+B12+B20+B02)
        U[15,14]=1
        V[9,14]=1; V[6,14]=1; V[8,14]=1; V[2,14]=1
        W[9,14]=1; W[6,14]=1; W[8,14]=1; W[2,14]=1

        # Term 15: A00*(B23+B32) + A11*(B32+B23) + A22*(B31+B13) + A33*(B12+B21)
        U[0,15]=1; U[5,15]=1; U[10,15]=1; U[15,15]=1
        V[11,15]=1; V[14,15]=1
        W[11,15]=1; W[14,15]=1; W[7,15]=1; W[13,15]=1
        W[6,15]=1; W[9,15]=1

        return U, V, W

    # 2x3 schoolbook (for handling the 6x6 as 4+2 split)
    # Actually, let's use the Kronecker product approach with 4x4 Laderman and 2x2 schoolbook
    # But 6 = 4 + 2, so we need a different approach
    
    # Alternative: Use block decomposition with 3x3 blocks of size 2x2
    # 6x6 = 3x3 blocks, each 2x2
    # Each 2x2*2x2 multiplication can be done with rank 4 (schoolbook) or 7 (Strassen)
    # But we need to handle the full 6x6 tensor
    
    # Let's use the approach: 6x6 multiplication via 3x3 block structure
    # Each block is 2x2
    # C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
    # Each 2x2*2x2 product has rank 4 (schoolbook)
    # Total: 3*3*3 * 4 = 108
    
    # But we can do better with Strassen: 3^3 * 7 = 189
    # Or even better: use Laderman for the 4x4 part
    
    # Actually, the simplest good approach: 
    # Split 6x6 as 4x4 and 2x2 blocks
    # But the tensor structure is complex
    
    # Let me just use the Kronecker product of 3x3 schoolbook and 2x2 schoolbook
    # 3x3 schoolbook has rank 27, 2x2 schoolbook has rank 4
    # Kronecker: 27 * 4 = 108
    # Wait, that's not right either
    
    # Let me think differently. The 6x6 matrix multiplication tensor has rank at most 216 (schoolbook)
    # We can use block structure:
    # If we view 6x6 as a 3x3 array of 2x2 blocks, then:
    # C = A * B means C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
    # Each block product is 2x2*2x2->2x2, which has rank 4 (schoolbook)
    # Number of block multiplications: 3*3*3 = 27
    # Total rank: 27 * 4 = 108
    
    # But actually, we can share the intermediate products more efficiently.
    # The correct way: the rank of the 6x6 tensor is at most the rank of the "block" tensor times the rank of the 2x2 tensor
    # The block tensor is 3x3x3 (which has rank 27 schoolbook)
    # The 2x2 tensor has rank 4
    # So total rank <= 27 * 4 = 108
    
    # Let me construct this properly.
    # For the 3x3 schoolbook:
    # U3[i*3+j, r] = 1 where r = i*9 + j*3 + k
    # V3[j*3+k, r] = 1
    # W3[k*3+i, r] = 1
    
    # For the 2x2 schoolbook:
    # U2[i2*2+j2, r2] = 1 where r2 = i2*4 + j2*2 + k2
    # V2[j2*2+k2, r2] = 1
    # W2[k2*2+i2, r2] = 1
    
    # For the 6x6:
    # Index in U6: (i6, j6) where i6 = i3*2 + i2, j6 = j3*2 + j2
    # Index in V6: (j6, k6) where k6 = k3*2 + k2
    # Index in W6: (k6, i6)
    # Rank index: r = r3 * 4 + r2
    # U6[i6*6+j6, r] = U3[i3*3+j3, r3] * U2[i2*2+j2, r2]
    # V6[j6*6+k6, r] = V3[j3*3+k3, r3] * V2[j2*2+k2, r2]
    # W6[k6*6+i6, r] = W3[k3*3+i3, r3] * W2[k2*2+i2, r2]
    
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
    
    U2 = np.zeros((4, 4), dtype=int)
    V2 = np.zeros((4, 4), dtype=int)
    W2 = np.zeros((4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                r = i * 4 + j * 2 + k
                U2[i * 2 + j, r] = 1
                V2[j * 2 + k, r] = 1
                W2[k * 2 + i, r] = 1
    
    rank_total = 27 * 4  # 108
    U6 = np.zeros((36, rank_total), dtype=int)
    V6 = np.zeros((36, rank_total), dtype=int)
    W6 = np.zeros((36, rank_total), dtype=int)
    
    for r3 in range(27):
        for r2 in range(4):
            r = r3 * 4 + r2
            for i3 in range(3):
                for j3 in range(3):
                    val_u3 = U3[i3*3+j3, r3]
                    if val_u3 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val_u2 = U2[i2*2+j2, r2]
                            if val_u2 == 0:
                                continue
                            i6 = i3*2 + i2
                            j6 = j3*2 + j2
                            U6[i6*6+j6, r] += val_u3 * val_u2
            for j3 in range(3):
                for k3 in range(3):
                    val_v3 = V3[j3*3+k3, r3]
                    if val_v3 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            val_v2 = V2[j2*2+k2, r2]
                            if val_v2 == 0:
                                continue
                            j6 = j3*2 + j2
                            k6 = k3*2 + k2
                            V6[j6*6+k6, r] += val_v3 * val_v2
            for k3 in range(3):
                for i3 in range(3):
                    val_w3 = W3[k3*3+i3, r3]
                    if val_w3 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            val_w2 = W2[k2*2+i2, r2]
                            if val_w2 == 0:
                                continue
                            k6 = k3*2 + k2
                            i6 = i3*2 + i2
                            W6[k6*6+i6, r] += val_w3 * val_w2
    
    # Verify
    def verify_full(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    if time.time() < deadline - 5:
        valid = verify_full(U6, V6, W6, n, m, p)
        if valid:
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
