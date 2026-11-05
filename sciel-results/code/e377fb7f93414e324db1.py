import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_fallback():
        """Strassen(2x2) x Schoolbook(3x3) -> Rank 189"""
        U2 = np.zeros((4, 7), dtype=int)
        V2 = np.zeros((4, 7), dtype=int)
        W2 = np.zeros((4, 7), dtype=int)
        
        # Strassen 2x2
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b21-b22)
        # m4 = a22(b12-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # 2x2 indices: U[0,0]=a11, U[0,1]=a12, U[1,0]=a21, U[1,1]=a22
        # V[0,0]=b11, V[0,1]=b12, V[1,0]=b21, V[1,1]=b22
        # W[0,0]=c11, W[0,1]=c12, W[1,0]=c21, W[1,1]=c22
        
        # r=0: m1
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        # c11 += m1, c22 += m1
        # Wait, standard Strassen:
        # c11 = m1 + m4 - m5 - m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m7
        
        # Let's re-verify the indices for the fallback to be safe, though it's the existing code.
        # The existing code in the prompt was:
        # U2[0, 0] = 1; U2[3, 0] = 1  (a11, a22)
        # V2[0, 0] = 1; V2[3, 0] = 1  (b11, b22)
        # W2[0, 0] = 1; W2[3, 0] = 1  (c11, c22)
        # This corresponds to m1 = (a11+a22)(b11+b22).
        # c11 gets +m1, c22 gets +m1. Correct.
        
        # r=1: m2 = (a21+a22)b11
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1
        # c21 += m2, c22 -= m2. Correct.
        
        # r=2: m3 = a11(b21-b22)
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1
        # c12 += m3, c22 += m3. Correct.
        
        # r=3: m4 = a22(b12-b11)
        U2[3, 3] = 1
        V2[0, 3] = -1; V2[2, 3] = 1
        W2[0, 3] = 1; W2[1, 3] = 1
        # c11 += m4, c21 += m4. Correct.
        
        # r=4: m5 = (a11+a12)b22
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1
        # c11 -= m5, c12 += m5. Correct.
        
        # r=5: m6 = (a21-a11)(b11+b12)
        # This term is usually not used in the 7-term Strassen, it's the 8th term in some variants?
        # No, Strassen uses 7 multiplications.
        # Let's check the standard 7 terms again.
        # 1. (a11+a22)(b11+b22)
        # 2. (a21+a22)b11
        # 3. a11(b21-b22)
        # 4. a22(b12-b11)
        # 5. (a11+a12)b22
        # 6. (a21-a11)(b11+b12)
        # 7. (a12-a22)(b21+b22)
        
        # The code above has:
        # r=5: U2[0, 5] = -1; U2[2, 5] = 1 -> (a21-a11)
        # V2[0, 5] = 1; V2[1, 5] = 1 -> (b11+b12)
        # W2[3, 5] = 1 -> c22 += m6?
        # Standard Strassen: c22 = m1 - m2 + m3 + m7. m6 is not in c22.
        # Wait, c21 = m2 + m4. c12 = m3 + m5. c11 = m1 + m4 - m5 - m7.
        # Where does m6 go?
        # Ah, the 7th term is often defined differently.
        # Let's trust the existing code for the fallback as it scored 1.1429 (rank 189).
        # I will just copy the existing fallback logic to ensure it works.
        
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

    def get_bjorcker_3x3():
        """Construct and verify a rank-20 3x3 decomposition."""
        rank3 = 20
        U3 = np.zeros((9, rank3), dtype=float)
        V3 = np.zeros((9, rank3), dtype=float)
        W3 = np.zeros((9, rank3), dtype=float)
        
        def ui(i, j): return i*3 + j
        def vi(j, k): return j*3 + k
        def wi(k, i): return k*3 + i
        
        # 9 terms for the "diagonal" A[i,j]*B[j,k] where i==k
        # These handle C[i,i] partially
        for i in range(3):
            for j in range(3):
                r = i*3 + j
                U3[ui(i,j), r] = 1
                V3[vi(j,i), r] = 1
                W3[wi(i,i), r] = 1
        
        # 9 terms for off-diagonal C[i,k] where i!=k
        # C[0,1] = a00*b01 + a01*b11 + a02*b21
        # C[1,0] = a10*b00 + a11*b10 + a12*b20
        # etc.
        # Use sum/diff tricks to reduce rank
        
        # Term 9: (a00+a01)*(b01+b10) -> contributes to c00, c01, c10, c11
        U3[ui(0,0),9]=1; U3[ui(0,1),9]=1
        V3[vi(0,1),9]=1; V3[vi(1,0),9]=1
        W3[wi(0,0),9]=-1; W3[wi(0,1),9]=1; W3[wi(1,0),9]=1; W3[wi(1,1),9]=-1
        
        # Term 10: (a00+a02)*(b02+b20) -> c00, c02, c20, c22
        U3[ui(0,0),10]=1; U3[ui(0,2),10]=1
        V3[vi(0,2),10]=1; V3[vi(2,0),10]=1
        W3[wi(0,0),10]=-1; W3[wi(0,2),10]=1; W3[wi(2,0),10]=1; W3[wi(2,2),10]=-1
        
        # Term 11: (a10+a11)*(b11+b10) -> c10, c11
        U3[ui(1,0),11]=1; U3[ui(1,1),11]=1
        V3[vi(1,1),11]=1; V3[vi(1,0),11]=1
        W3[wi(1,0),11]=-1; W3[wi(1,1),11]=-1
        
        # Term 12: (a11+a12)*(b12+b21) -> c11, c12, c21, c22
        U3[ui(1,1),12]=1; U3[ui(1,2),12]=1
        V3[vi(1,2),12]=1; V3[vi(2,1),12]=1
        W3[wi(1,1),12]=-1; W3[wi(1,2),12]=1; W3[wi(2,1),12]=1; W3[wi(2,2),12]=-1
        
        # Term 13: (a20+a21)*(b21+b10) -> c20, c21
        U3[ui(2,0),13]=1; U3[ui(2,1),13]=1
        V3[vi(2,1),13]=1; V3[vi(1,0),13]=1
        W3[wi(2,0),13]=-1; W3[wi(2,1),13]=-1
        
        # Term 14: (a21+a22)*(b22+b21) -> c21, c22
        U3[ui(2,1),14]=1; U3[ui(2,2),14]=1
        V3[vi(2,2),14]=1; V3[vi(2,1),14]=1
        W3[wi(2,1),14]=-1; W3[wi(2,2),14]=-1
        
        # Term 15: (a01+a02)*(b12+b22) -> c02
        U3[ui(0,1),15]=1; U3[ui(0,2),15]=1
        V3[vi(1,2),15]=1; V3[vi(2,2),15]=1
        W3[wi(2,0),15]=1
        
        # Term 16: (a10+a12)*(b02+b20) -> c10, c12, c20
        U3[ui(1,0),16]=1; U3[ui(1,2),16]=1
        V3[vi(0,2),16]=1; V3[vi(2,0),16]=1
        W3[wi(2,1),16]=1; W3[wi(0,1),16]=1
        
        # Term 17: (a00+a11)*(b00+b11) -> c00, c11
        U3[ui(0,0),17]=1; U3[ui(1,1),17]=1
        V3[vi(0,0),17]=1; V3[vi(1,1),17]=1
        W3[wi(0,0),17]=-1; W3[wi(1,1),17]=-1
        
        # Term 18: (a00+a22)*(b00+b22) -> c00, c22
        U3[ui(0,0),18]=1; U3[ui(2,2),18]=1
        V3[vi(0,0),18]=1; V3[vi(2,2),18]=1
        W3[wi(0,0),18]=-1; W3[wi(2,2),18]=-1
        
        # Term 19: (a01+a20)*(b10+b01) -> c01, c10, c20, c21
        U3[ui(0,1),19]=1; U3[ui(2,0),19]=1
        V3[vi(1,0),19]=1; V3[vi(0,1),19]=1
        W3[wi(0,1),19]=1; W3[wi(1,0),19]=1; W3[wi(2,0),19]=1; W3[wi(1,2),19]=1
        
        return U3, V3, W3

    def try_bjorcker_solution():
        """Build rank-140 solution using Strassen(2x2) x Bjorcker(3x3)."""
        if time.time() > deadline - 10:
            return None
        
        U3, V3, W3 = get_bjorcker_3x3()
        rank3 = U3.shape[1]
        
        # Verify 3x3 decomposition using vectorized numpy (fast)
        U3_3d = U3.reshape(3, 3, rank3)
        V3_3d = V3.reshape(3, 3, rank3)
        W3_3d = W3.reshape(3, 3, rank3)
        
        T3 = np.zeros((3,3,3))
        for r in range(rank3):
            T3 += np.einsum('ij,jk,ki->ijk', U3_3d[:,:,r], V3_3d[:,:,r], W3_3d[:,:,r])
        
        if not np.allclose(T3, 1.0, atol=1e-9):
            return None
        
        # Strassen 2x2
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

        rank = 7 * rank3
        U6 = np.zeros((36, rank), dtype=float)
        V6 = np.zeros((36, rank), dtype=float)
        W6 = np.zeros((36, rank), dtype=float)
        
        for r2 in range(7):
            for r3 in range(rank3):
                r = r2 * rank3 + r3
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
        
        # Verify full 6x6 decomposition using vectorized numpy (fast)
        if time.time() > deadline - 5:
            return None
        
        # T6[i,j,k] = sum_r U6[i*6+j,r] * V6[j*6+k,r] * W6[k*6+i,r]
        U6_3d = U6.reshape(6, 6, rank)
        V6_3d = V6.reshape(6, 6, rank)
        W6_3d = W6.reshape(6, 6, rank)
        
        T6 = np.zeros((6,6,6))
        for r in range(rank):
            T6 += np.einsum('ij,jk,ki->ijk', U6_3d[:,:,r], V6_3d[:,:,r], W6_3d[:,:,r])
        
        if not np.allclose(T6, 1.0, atol=1e-6):
            return None
        
        # Check entries are integer or half-integer
        for mat in [U6, V6, W6]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return None
        
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Try the better Bjorcker-based solution first (if it passes verification)
    if time.time() < deadline - 30:
        result = try_bjorcker_solution()
        if result is not None:
            return result
    
    # Fall back to Strassen x Schoolbook (rank 189)
    return get_fallback()
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
