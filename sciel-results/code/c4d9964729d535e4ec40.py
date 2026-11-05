import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    t_start = time.time()
    
    # Try multiple Strassen 2x2 formula variants
    def make_strassen_2x2(variant):
        U2 = np.zeros((4, 7), dtype=int)
        V2 = np.zeros((4, 7), dtype=int)
        W2 = np.zeros((4, 7), dtype=int)
        
        if variant == 0:
            # Variant 0
            U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1
            U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; V2[1,1]=1
            U2[0,2]=1; U2[1,2]=1; V2[2,2]=1; V2[3,2]=1
            U2[0,3]=1; V2[0,3]=1; V2[2,3]=-1
            U2[3,4]=1; V2[1,4]=1; V2[3,4]=-1
            U2[0,5]=1; U2[2,5]=1; V2[3,5]=1; V2[1,5]=-1
            U2[3,6]=1; U2[1,6]=1; V2[0,6]=1; V2[2,6]=-1
            W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
            W2[1,2]=1; W2[1,4]=1; W2[1,0]=-1; W2[1,5]=1
            W2[2,1]=1; W2[2,3]=1; W2[2,6]=-1; W2[2,2]=1
            W2[3,0]=1; W2[3,1]=-1; W2[3,4]=1; W2[3,5]=1
        elif variant == 1:
            # M1=(a11+a22)(b11+b22), M2=(a21+a22)b11, M3=a11(b12-b22)
            # M4=a22(b21-b11), M5=(a11+a12)b22, M6=(a21-a11)(b11+b12)
            # M7=(a12-a22)(b21+b22)
            # c11=M1-M2+M4-M5, c12=M3+M5, c21=M1+M2, c22=M4-M1+M3+M6
            U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1
            U2[2,1]=1; U2[3,1]=1; V2[0,1]=1
            U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1
            U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1
            U2[0,4]=1; U2[1,4]=1; V2[3,4]=1
            U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1
            U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1
            W2[0,0]=1; W2[0,1]=-1; W2[0,3]=1; W2[0,4]=-1
            W2[1,2]=1; W2[1,4]=1
            W2[2,0]=1; W2[2,1]=1
            W2[3,3]=1; W2[3,0]=-1; W2[3,2]=1; W2[3,5]=1
        elif variant == 2:
            # Another variant
            U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1
            U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; V2[1,1]=1
            U2[0,2]=1; U2[1,2]=1; V2[2,2]=1; V2[3,2]=1
            U2[0,3]=1; V2[0,3]=1; V2[2,3]=-1
            U2[3,4]=1; V2[1,4]=1; V2[3,4]=-1
            U2[0,5]=1; U2[2,5]=1; V2[3,5]=1; V2[1,5]=-1
            U2[3,6]=1; U2[1,6]=1; V2[0,6]=1; V2[2,6]=-1
            W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
            W2[1,2]=1; W2[1,4]=1; W2[1,0]=-1; W2[1,5]=1
            W2[2,1]=1; W2[2,3]=1; W2[2,6]=-1; W2[2,2]=1
            W2[3,0]=1; W2[3,1]=-1; W2[3,4]=1; W2[3,5]=1
        
        return U2, V2, W2
    
    # Test each variant
    A_test = np.array([[1,2],[3,4]], dtype=float)
    B_test = np.array([[5,6],[7,8]], dtype=float)
    C_test = A_test @ B_test
    
    valid_variant = -1
    for v in range(3):
        U2, V2, W2 = make_strassen_2x2(v)
        C_calc = np.zeros((2,2), dtype=float)
        for c in range(7):
            p_c = sum(U2[i,c]*A_test.ravel()[i] for i in range(4)) * \
                  sum(V2[j,c]*B_test.ravel()[j] for j in range(4))
            for k in range(4):
                C_calc.ravel()[k] += W2[k,c] * p_c
        if np.allclose(C_calc, C_test, atol=1e-10):
            valid_variant = v
            break
    
    if valid_variant == -1:
        # Fallback: schoolbook for everything, rank 80
        rank = 80
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(20)]
        W = [[0]*rank for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U2, V2, W2 = make_strassen_2x2(valid_variant)
    
    # 4x4x4 Strassen (rank 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    A_combos = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1, 1: 1},
        {0: 1}, {3: 1}, {0: 1, 2: 1}, {3: 1, 1: 1}
    ]
    B_combos = [
        {0: 1, 3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1},
        {0: 1, 2: -1}, {1: 1, 3: -1}, {3: 1, 1: -1}, {0: 1, 2: -1}
    ]
    C_combos = [
        {0: 1, 3: 1, 4: -1, 6: 1},
        {2: 1, 4: 1, 0: -1, 5: 1},
        {1: 1, 3: 1, 6: -1, 2: 1},
        {0: 1, 1: -1, 4: 1, 5: 1}
    ]
    
    for m in range(7):
        for s in range(7):
            cg = m * 7 + s
            for block_a, coeff_a in A_combos[m].items():
                for li in range(4):
                    U4[block_a*4+li, cg] += coeff_a * U2[li, s]
            for block_b, coeff_b in B_combos[m].items():
                for lj in range(4):
                    V4[block_b*4+lj, cg] += coeff_b * V2[lj, s]
            for cb in range(4):
                coeff_c = C_combos[cb].get(m, 0)
                if coeff_c != 0:
                    for lk in range(4):
                        W4[cb*4+lk, cg] += coeff_c * W2[lk, s]
    
    # Verify 4x4
    A4 = np.random.randint(-3, 4, (4,4)).astype(float)
    B4 = np.random.randint(-3, 4, (4,4)).astype(float)
    C4_exp = A4 @ B4
    C4_calc = np.zeros((4,4), dtype=float)
    for cg in range(49):
        u = sum(U4[i,cg]*A4.ravel()[i] for i in range(16))
        v = sum(V4[j,cg]*B4.ravel()[j] for j in range(16))
        for k in range(16):
            C4_calc.ravel()[k] += W4[k,cg]*u*v
    if not np.allclose(C4_calc, C4_exp, atol=1e-10):
        # Fallback schoolbook
        rank = 80
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(20)]
        W = [[0]*rank for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Full 4x4x5
    rank = 49 + 16
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((20, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    
    for cg in range(49):
        U[:, cg] = U4[:, cg]
        for j in range(4):
            for k in range(4):
                V[j*5+k, cg] = V4[j*4+k, cg]
        for i in range(4):
            for k in range(4):
                W[k*4+i, cg] = W4[i*4+k, cg]
    
    for i in range(4):
        for j in range(4):
            c = 49 + i*4 + j
            U[i*4+j, c] = 1
            V[j*5+4, c] = 1
            W[16+i, c] = 1
    
    # Final verification
    A_full = np.random.randint(-3, 4, (4,4)).astype(float)
    B_full = np.random.randint(-3, 4, (4,5)).astype(float)
    C_exp = A_full @ B_full
    C_calc = np.zeros((4,5), dtype=float)
    for r in range(rank):
        u = sum(U[i*4+j, r]*A_full[i,j] for i in range(4) for j in range(4))
        v = sum(V[j*5+k, r]*B_full[j,k] for j in range(4) for k in range(5))
        for i in range(4):
            for k in range(5):
                C_calc[i,k] += W[k*4+i, r]*u*v
    
    if not np.allclose(C_calc, C_exp, atol=1e-10):
        # Fallback
        rank = 80
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(20)]
        W = [[0]*rank for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
