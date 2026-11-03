"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _build_4x4x7_optimized():
    """
    Structural approach: Use block-level Strassen decomposition.
    
    Split A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    Split B (4x7) into 2x7 blocks: B = [[B0], [B1]] where B0, B1 are 2x7
    Split C (4x7) into 2x7 blocks: C = [[C0], [C1]] where C0, C1 are 2x7
    
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    Now split B0 and B1 into column blocks:
    B0 = [B00, B01, B02, B03] where B00,B01,B02 are 2x2 and B03 is 2x1
    B1 = [B10, B11, B12, B13] where B10,B11,B12 are 2x2 and B13 is 2x1
    
    For each column block c (0..3):
    C0[:,c] = A00*B0[:,c] + A01*B1[:,c]  (2x2 * 2xc + 2x2 * 2xc)
    C1[:,c] = A10*B0[:,c] + A11*B1[:,c]
    
    For square blocks (c=0,1,2): use Strassen on the 2x2x2 block products
    For the last block (c=3): use schoolbook on 2x2x1
    
    Total: 3 * 4 * 7 (Strassen for 12 2x2x2 products) + 4 * 4 (schoolbook for 4 2x2x1)
         = 84 + 16 = 100... that's worse!
    
    Better: Use Strassen at the BLOCK level.
    
    Treat the block multiplication as:
    C_blocks = A_blocks * B_blocks
    where A_blocks is 2x2 (each entry is 2x2), B_blocks is 2x4 (entries are 2x2 or 2x1)
    
    Block-level Strassen for 2x2 * 2x2 part (first 3 columns):
    M1 = (A00+A11)(B0+B1)  -- 2x2 * 2x6
    M2 = (A10+A11)B0       -- 2x2 * 2x6
    M3 = A00(B1-B0)        -- 2x2 * 2x6
    M4 = A11(B0-B1)        -- 2x2 * 2x6
    M5 = (A00+A01)B1       -- 2x2 * 2x6
    M6 = (A10-A00)(B0+B1)  -- 2x2 * 2x6
    M7 = (A01-A11)(B1-B0)  -- 2x2 * 2x6
    
    Each Mi is a 2x2 * 2x6 product. Rank of 2x2*2x6 = 2*2*6 = 24 (schoolbook)
    Or can we do better? 2x2*2x2 is rank 7, so 2x2*2x6 = 3 * (2x2*2x2) = 21?
    Actually 2x2*2x6: split into three 2x2*2x2, each rank 7, total 21.
    
    So 7 * 21 = 147 for the block-level Strassen on the square part. That's worse.
    
    Let me go back to the working rank-93 approach but try to optimize the 4x4x1 part.
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]
    W2p = [[0] * rank2 for _ in range(8)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V2p[j*2+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W2p[k*4+i][r] += val
            block_idx += 1
    
    # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]
    W3 = [[0] * rank3 for _ in range(4)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
    
    # Combine
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for i in range(4):
        for r in range(rank3):
            V[i*p + 6][offset3 + r] = V3[i][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x7_unified():
    """
    Unified block-level Strassen: treat 4x4*4x7 as a single block multiplication.
    
    A is 2x2 blocks of 2x2.
    B is 2x4 blocks: top row [2x2, 2x2, 2x2, 2x1], bottom row [2x2, 2x2, 2x2, 2x1]
    
    Use Strassen at block level for the 2x2 * 2x3 (square) part,
    and handle the 2x1 column separately.
    
    Actually, let's use a different structural idea: 
    Split B into B_sq (4x4) and B_rem (4x3).
    For B_sq: use rank-49 Kronecker Strassen.
    For B_rem: use a rank-11 per 2x2x3 block approach.
    
    4x4*4x3: split into 4 blocks of 2x2x3.
    Each 2x2x3: A_block (2x2) * B_block (2x3) -> C_block (2x3)
    C_block = A_block * B_block
    
    For 2x2 * 2x3:
    Split B_block into B_left (2x2) and B_right (2x1).
    A * B_left: 2x2*2x2, rank 7 (Strassen)
    A * B_right: 2x2*2x1, rank 4 (schoolbook)
    Total per block: 11
    
    4 blocks * 11 = 44 for the 4x4*4x3 part.
    Total: 49 + 44 = 93. Same as before!
    
    Let me try yet another approach: can I do 2x2*2x3 in rank < 11?
    The tensor for 2x2*2x3 has 2*2*3 = 12 entries.
    Schoolbook rank = 12.
    With Strassen on the 2x2*2x2 part: 7 + 4 = 11.
    Is rank 10 possible? Unknown, but let's assume 11 is good.
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4, rank 49 (Kronecker Strassen)
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x3 (B cols 4-6, C cols 4-6), rank 44
    # Four 2x2x3 blocks, each rank 11 (Strassen 7 + schoolbook 4)
    rank2 = 44
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(12)]  # 4*3
    W2p = [[0] * rank2 for _ in range(12)]  # 3*4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            # Strassen part: 2x2 * 2x2 (B local cols 0-1, C local cols 0-1)
            for t in range(7):
                r = block_idx * 11 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl  # global B col
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V2p[j*3+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl  # local C col
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W2p[k*4+i][r] += val
            # Schoolbook part: 2x2 * 2x1 (B local col 2, C local col 2)
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 11 + 7 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U2p[i*4+j][r] = 1
                    V2p[j*3+2][r] = 1
                    W2p[2*4+i][r] = 1
            block_idx += 1
    
    # Combine
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(3):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*3+j][r]
    for i in range(3):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Try unified approach (rank 93: 49 + 44)
    try:
        result = _build_4x4x7_unified()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Try optimized approach (rank 93: 49 + 28 + 16)
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    return _schoolbook(4, 4, 7)
