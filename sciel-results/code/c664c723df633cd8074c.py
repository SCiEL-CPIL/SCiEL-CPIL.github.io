"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2():
        """Return Strassen rank-7 decomposition for 2x2x2.
        U: 4 rows (i*m+j = i*2+j), V: 4 rows (j*p+k = j*2+k), W: 4 rows (k*n+i = k*2+i)
        """
        # Strassen 7 terms for 2x2x2
        # t1 = (a00+a11)(b00+b11) -> c00 += t1, c01 -= t1, c10 -= t1, c11 += t1
        # t2 = (a10+a11)b00 -> c10 += t2, c11 += t2
        # t3 = a00(b01-b11) -> c00 += t3, c01 += t3
        # t4 = a11(b10-b00) -> c10 += t4, c11 += t4
        # t5 = (a00+a01)b11 -> c00 -= t5, c11 += t5
        # t6 = (a10-a00)(b00+b01) -> c10 -= t6, c11 += t6
        # t7 = (a01-a11)(b10+b11) -> c01 += t7, c11 -= t7

        # U rows: (i,j) -> i*2+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V rows: (j,k) -> j*2+k: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W rows: (k,i) -> k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # t1: U: a00=1, a11=1; V: b00=1, b11=1; W: c00+=1, c01-=1, c10-=1, c11+=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = -1; W[2][0] = -1; W[3][0] = 1

        # t2: U: a10=1, a11=1; V: b00=1; W: c10+=1, c11+=1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1

        # t3: U: a00=1; V: b01=1, b11=-1; W: c00+=1, c01+=1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[1][2] = 1

        # t4: U: a11=1; V: b10=1, b00=-1; W: c10+=1, c11+=1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[3][3] = 1

        # t5: U: a00=1, a01=1; V: b11=1; W: c00-=1, c11+=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1

        # t6: U: a10=1, a00=-1; V: b00=1, b01=1; W: c10-=1, c11+=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = -1; W[3][5] = 1

        # t7: U: a01=1, a11=-1; V: b10=1, b11=1; W: c01+=1, c11-=1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[1][6] = 1; W[3][6] = -1

        return U, V, W, 7

    def schoolbook_block(n_b, m_b, p_b):
        """Schoolbook decomposition for n_b x m_b x p_b block."""
        rank = n_b * m_b * p_b
        U = [[0] * rank for _ in range(n_b * m_b)]
        V = [[0] * rank for _ in range(m_b * p_b)]
        W = [[0] * rank for _ in range(p_b * n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j][col] = 1
                    V[j * p_b + k][col] = 1
                    W[k * n_b + i][col] = 1
                    col += 1
        return U, V, W, rank

    def block_2x4x4():
        """2x4x4 = (2x2x2) + (2x2x2), rank <= 14."""
        U1, V1, W1, r1 = strassen_2x2x2()
        U2, V2, W2, r2 = strassen_2x2x2()
        rank = r1 + r2
        U = [[0] * rank for _ in range(8)]  # 2*4 = 8 rows
        V = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
        W = [[0] * rank for _ in range(8)]  # 4*2 = 8 rows

        # First 2x2x2: A_left (2x2) * B_top (2x2)
        # A_left: rows (i, j) for i in 0,1, j in 0,1 -> global (i, j)
        # B_top: rows (j, k) for j in 0,1, k in 0,1 -> global (j, k)
        # Output: (k, i) for k in 0,1, i in 0,1 -> global (k, i)
        for c in range(r1):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + j][c] = U1[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    V[j * 4 + k][c] = V1[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    W[k * 2 + i][c] = W1[k * 2 + i][c]

        # Second 2x2x2: A_right (2x2) * B_bottom (2x2)
        # A_right: rows (i, j) for i in 0,1, j in 2,3 -> global (i, j)
        # B_bottom: rows (j, k) for j in 2,3, k in 0,1 -> global (j, k)
        # Output: (k, i) for k in 2,3, i in 0,1 -> global (k, i)
        for c in range(r2):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (j + 2)][c + r1] = U2[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    V[(j + 2) * 4 + k][c + r1] = V2[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    W[(k + 2) * 2 + i][c + r1] = W2[k * 2 + i][c]

        return U, V, W, rank

    def block_2x4x2():
        """2x4x2 = (2x2x2) + (2x2x2), rank <= 14."""
        U1, V1, W1, r1 = strassen_2x2x2()
        U2, V2, W2, r2 = strassen_2x2x2()
        rank = r1 + r2
        U = [[0] * rank for _ in range(8)]  # 2*4 = 8 rows
        V = [[0] * rank for _ in range(8)]  # 4*2 = 8 rows
        W = [[0] * rank for _ in range(4)]  # 2*2 = 4 rows

        # First: A_left (2x2) * B_top (2x2), output to C top-left (2x2)
        for c in range(r1):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + j][c] = U1[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    V[j * 2 + k][c] = V1[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    W[k * 2 + i][c] = W1[k * 2 + i][c]

        # Second: A_right (2x2) * B_bottom (2x2), output to C bottom-left... 
        # Wait, 2x4x2: A is 2x4, B is 4x2, C is 2x2
        # A = [A1 | A2] where A1, A2 are 2x2
        # B = [B1; B2] where B1, B2 are 2x2
        # C = A1*B1 + A2*B2
        # Both contribute to the same 2x2 output C
        for c in range(r2):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (j + 2)][c + r1] = U2[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    V[(j + 2) * 2 + k][c + r1] = V2[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    W[k * 2 + i][c + r1] = W2[k * 2 + i][c]

        return U, V, W, rank

    def verify_full(U, V, W, rank):
        """Verify full decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Partition: 3 = [0,1]+[2], 5 = [0,1,2,3]+[4], 6 = [0,1,2,3]+[4,5]
    i_parts = [[0, 1], [2]]
    j_parts = [[0, 1, 2, 3], [4]]
    k_parts = [[0, 1, 2, 3], [4, 5]]

    # Get block decompositions
    block_decomps = {}
    
    # (2,4,4)
    block_decomps[(2,4,4)] = block_2x4x4()
    # (2,4,2)
    block_decomps[(2,4,2)] = block_2x4x2()
    # (2,1,4) - schoolbook
    block_decomps[(2,1,4)] = schoolbook_block(2, 1, 4)
    # (2,1,2) - schoolbook
    block_decomps[(2,1,2)] = schoolbook_block(2, 1, 2)
    # (1,4,4) - schoolbook
    block_decomps[(1,4,4)] = schoolbook_block(1, 4, 4)
    # (1,4,2) - schoolbook
    block_decomps[(1,4,2)] = schoolbook_block(1, 4, 2)
    # (1,1,4) - schoolbook
    block_decomps[(1,1,4)] = schoolbook_block(1, 1, 4)
    # (1,1,2) - schoolbook
    block_decomps[(1,1,2)] = schoolbook_block(1, 1, 2)

    # Combine all blocks
    total_rank = sum(decomp[3] for decomp in block_decomps.values())
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for ig in i_parts:
        for jg in j_parts:
            for kg in k_parts:
                si, sj, sk = len(ig), len(jg), len(kg)
                U_b, V_b, W_b, r_b = block_decomps[(si, sj, sk)]
                for c in range(r_b):
                    for li, gi in enumerate(ig):
                        for lj, gj in enumerate(jg):
                            U[gi * m + gj][col + c] = U_b[li * sj + lj][c]
                    for lj, gj in enumerate(jg):
                        for lk, gk in enumerate(kg):
                            V[gj * p + gk][col + c] = V_b[lj * sk + lk][c]
                    for lk, gk in enumerate(kg):
                        for li, gi in enumerate(ig):
                            W[gk * n + gi][col + c] = W_b[lk * si + li][c]
                col += r_b

    # Verify
    if verify_full(U, V, W, total_rank):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    U, V, W, _ = schoolbook_block(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
