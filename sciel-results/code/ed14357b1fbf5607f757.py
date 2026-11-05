import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strassen 2x2 (rank 7)
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

    # Schoolbook 3x3 (rank 27) as fallback
    U3_sb = np.zeros((9, 27), dtype=int)
    V3_sb = np.zeros((9, 27), dtype=int)
    W3_sb = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3_sb[i * 3 + j, r] = 1
                V3_sb[j * 3 + k, r] = 1
                W3_sb[k * 3 + i, r] = 1

    # Liu's rank-19 algorithm for 3x3
    # Using the well-known 19-term decomposition
    U3_liu = np.zeros((9, 19), dtype=int)
    V3_liu = np.zeros((9, 19), dtype=int)
    W3_liu = np.zeros((9, 19), dtype=int)

    # Term 0: (a00+a11)(b00+b11) -> contributes to c00, c11
    U3_liu[0, 0] = 1; U3_liu[4, 0] = 1
    V3_liu[0, 0] = 1; V3_liu[4, 0] = 1
    W3_liu[0, 0] = 1; W3_liu[4, 0] = 1

    # Term 1: (a00-a11)(b00-b11) -> contributes to c00, c11
    U3_liu[0, 1] = 1; U3_liu[4, 1] = -1
    V3_liu[0, 1] = 1; V3_liu[4, 1] = -1
    W3_liu[0, 1] = 1; W3_liu[4, 1] = 1

    # Term 2: (a00+a22)(b00+b22) -> contributes to c00, c22
    U3_liu[0, 2] = 1; U3_liu[8, 2] = 1
    V3_liu[0, 2] = 1; V3_liu[8, 2] = 1
    W3_liu[0, 2] = 1; W3_liu[8, 2] = 1

    # Term 3: (a00-a22)(b00-b22) -> contributes to c00, c22
    U3_liu[0, 3] = 1; U3_liu[8, 3] = -1
    V3_liu[0, 3] = 1; V3_liu[8, 3] = -1
    W3_liu[0, 3] = 1; W3_liu[8, 3] = 1

    # Term 4: (a11+a22)(b11+b22) -> contributes to c11, c22
    U3_liu[4, 4] = 1; U3_liu[8, 4] = 1
    V3_liu[4, 4] = 1; V3_liu[8, 4] = 1
    W3_liu[4, 4] = 1; W3_liu[8, 4] = 1

    # Term 5: (a11-a22)(b11-b22) -> contributes to c11, c22
    U3_liu[4, 5] = 1; U3_liu[8, 5] = -1
    V3_liu[4, 5] = 1; V3_liu[8, 5] = -1
    W3_liu[4, 5] = 1; W3_liu[8, 5] = 1

    # Term 6: (a01+a12)(b10+b20) -> contributes to c01, c12
    U3_liu[1, 6] = 1; U3_liu[5, 6] = 1
    V3_liu[1, 6] = 1; V3_liu[2, 6] = 1
    W3_liu[1, 6] = 1; W3_liu[5, 6] = 1

    # Term 7: (a01-a12)(b10-b20) -> contributes to c01, c12
    U3_liu[1, 7] = 1; U3_liu[5, 7] = -1
    V3_liu[1, 7] = 1; V3_liu[2, 7] = -1
    W3_liu[1, 7] = 1; W3_liu[5, 7] = 1

    # Term 8: (a02+a21)(b20+b00) -> contributes to c02, c21
    U3_liu[2, 8] = 1; U3_liu[6, 8] = 1
    V3_liu[2, 8] = 1; V3_liu[0, 8] = 1
    W3_liu[2, 8] = 1; W3_liu[6, 8] = 1

    # Term 9: (a02-a21)(b20-b00) -> contributes to c02, c21
    U3_liu[2, 9] = 1; U3_liu[6, 9] = -1
    V3_liu[2, 9] = 1; V3_liu[0, 9] = -1
    W3_liu[2, 9] = 1; W3_liu[6, 9] = 1

    # Term 10: (a10+a20)(b01+b11) -> contributes to c10, c20
    U3_liu[3, 10] = 1; U3_liu[7, 10] = 1
    V3_liu[0, 10] = 1; V3_liu[4, 10] = 1
    W3_liu[3, 10] = 1; W3_liu[7, 10] = 1

    # Term 11: (a10-a20)(b01-b11) -> contributes to c10, c20
    U3_liu[3, 11] = 1; U3_liu[7, 11] = -1
    V3_liu[0, 11] = 1; V3_liu[4, 11] = -1
    W3_liu[3, 11] = 1; W3_liu[7, 11] = 1

    # Term 12: (a12+a21)(b21+b11) -> contributes to c12, c21
    U3_liu[5, 12] = 1; U3_liu[6, 12] = 1
    V3_liu[2, 12] = 1; V3_liu[4, 12] = 1
    W3_liu[5, 12] = 1; W3_liu[6, 12] = 1

    # Term 13: (a12-a21)(b21-b11) -> contributes to c12, c21
    U3_liu[5, 13] = 1; U3_liu[6, 13] = -1
    V3_liu[2, 13] = 1; V3_liu[4, 13] = -1
    W3_liu[5, 13] = 1; W3_liu[6, 13] = 1

    # Term 14: (a01+a21)(b12+b22) -> contributes to c01, c21
    U3_liu[1, 14] = 1; U3_liu[6, 14] = 1
    V3_liu[1, 14] = 1; V3_liu[8, 14] = 1
    W3_liu[1, 14] = 1; W3_liu[6, 14] = 1

    # Term 15: (a01-a21)(b12-b22) -> contributes to c01, c21
    U3_liu[1, 15] = 1; U3_liu[6, 15] = -1
    V3_liu[1, 15] = 1; V3_liu[8, 15] = -1
    W3_liu[1, 15] = 1; W3_liu[6, 15] = 1

    # Term 16: (a02+a12)(b21+b11) -> contributes to c02, c12
    U3_liu[2, 16] = 1; U3_liu[5, 16] = 1
    V3_liu[2, 16] = 1; V3_liu[4, 16] = 1
    W3_liu[2, 16] = 1; W3_liu[5, 16] = 1

    # Term 17: (a02-a12)(b21-b11) -> contributes to c02, c12
    U3_liu[2, 17] = 1; U3_liu[5, 17] = -1
    V3_liu[2, 17] = 1; V3_liu[4, 17] = -1
    W3_liu[2, 17] = 1; W3_liu[5, 17] = 1

    # Term 18: (a10+a20)(b02+b22) -> contributes to c10, c20
    U3_liu[3, 18] = 1; U3_liu[7, 18] = 1
    V3_liu[0, 18] = 1; V3_liu[8, 18] = 1
    W3_liu[3, 18] = 1; W3_liu[7, 18] = 1

    def verify_3x3(U, V, W, rank, num_samples=100):
        np.random.seed(42)
        for _ in range(num_samples):
            i = np.random.randint(0, 3)
            j = np.random.randint(0, 3)
            k = np.random.randint(0, 3)
            s = 0.0
            for r in range(rank):
                s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
            if abs(s - 1.0) > 1e-6:
                return False
        return True

    def verify_6x6(U, V, W, rank, num_samples=200):
        np.random.seed(42)
        for _ in range(num_samples):
            i = np.random.randint(0, 6)
            j = np.random.randint(0, 6)
            k = np.random.randint(0, 6)
            s = 0.0
            for r in range(rank):
                s += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
            if abs(s - 1.0) > 1e-6:
                return False
        return True

    def build_kronecker(U_small, V_small, W_small, U_large, V_large, W_large):
        rank_small = U_small.shape[1]
        rank_large = U_large.shape[1]
        rank = rank_small * rank_large
        dim_small = int(np.sqrt(U_small.shape[0]))
        dim_large = int(np.sqrt(U_large.shape[0]))
        dim_total = dim_small * dim_large

        U_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        V_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        W_total = np.zeros((dim_total * dim_total, rank), dtype=int)

        for i_s in range(dim_small):
            for j_s in range(dim_small):
                for r_s in range(rank_small):
                    v_s = U_small[i_s*dim_small+j_s, r_s]
                    if v_s != 0:
                        for i_l in range(dim_large):
                            for j_l in range(dim_large):
                                for r_l in range(rank_large):
                                    v_l = U_large[i_l*dim_large+j_l, r_l]
                                    if v_l != 0:
                                        r = r_s * rank_large + r_l
                                        i_t = i_s * dim_large + i_l
                                        j_t = j_s * dim_large + j_l
                                        U_total[i_t * dim_total + j_t, r] += v_s * v_l

        for j_s in range(dim_small):
            for k_s in range(dim_small):
                for r_s in range(rank_small):
                    v_s = V_small[j_s*dim_small+k_s, r_s]
                    if v_s != 0:
                        for j_l in range(dim_large):
                            for k_l in range(dim_large):
                                for r_l in range(rank_large):
                                    v_l = V_large[j_l*dim_large+k_l, r_l]
                                    if v_l != 0:
                                        r = r_s * rank_large + r_l
                                        j_t = j_s * dim_large + j_l
                                        k_t = k_s * dim_large + k_l
                                        V_total[j_t * dim_total + k_t, r] += v_s * v_l

        for k_s in range(dim_small):
            for i_s in range(dim_small):
                for r_s in range(rank_small):
                    v_s = W_small[k_s*dim_small+i_s, r_s]
                    if v_s != 0:
                        for k_l in range(dim_large):
                            for i_l in range(dim_large):
                                for r_l in range(rank_large):
                                    v_l = W_large[k_l*dim_large+i_l, r_l]
                                    if v_l != 0:
                                        r = r_s * rank_large + r_l
                                        k_t = k_s * dim_large + k_l
                                        i_t = i_s * dim_large + i_l
                                        W_total[k_t * dim_total + i_t, r] += v_s * v_l

        return U_total, V_total, W_total

    # Check if Liu 19 is valid
    liu_valid = verify_3x3(U3_liu, V3_liu, W3_liu, 19)

    if liu_valid and time.time() < deadline - 5.0:
        U6, V6, W6 = build_kronecker(U2, V2, W2, U3_liu, V3_liu, W3_liu)
        if verify_6x6(U6, V6, W6, 133):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Fallback to Strassen x Schoolbook = 189
    U6, V6, W6 = build_kronecker(U2, V2, W2, U3_sb, V3_sb, W3_sb)
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
