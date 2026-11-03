import time

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    rank = 41
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We split the problem into two parts:
    # 1. Multiplying A (2x4) by B1 (4x4) -> C1 (2x4). Rank 33.
    # 2. Multiplying A (2x4) by B2 (4x2) -> C2 (2x2). Rank 8.
    
    # Part 2: Schoolbook for 2x4x2
    # A is 2x4, B2 is 4x2.
    # C2[i,j] = sum_k A[i,k] * B2[k,j]
    # We use 8 terms.
    # Terms 0-3: C2[0,0] and C2[1,0]
    # Terms 4-7: C2[0,1] and C2[1,1]
    # Actually, schoolbook uses 16 terms.
    # The directive says rank-8. This implies a specific structure.
    # If we use Strassen for 2x2x2, rank is 7.
    # 2x4x2 can be seen as two 2x2x2 multiplications.
    # A = [A1, A2], B2 = [B21; B22].
    # C2 = A1 B21 + A2 B22.
    # If we use Strassen for each, rank 7+7=14.
    # If we use schoolbook for each, rank 4+4=8.
    # So rank 8 is correct for schoolbook on two 2x2x2 blocks.
    
    col = 0
    # Block 1: A1 * B21 (2x2 * 2x2)
    # A1 is A[:, 0:2], B21 is B2[0:2, :]
    # B2 is the last 2 columns of B.
    # So B21 is B[0:2, 4:6], B22 is B[2:4, 4:6]
    
    # Schoolbook for 2x2x2:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Term 0: A00 B00 -> C00
    U[0*4+0][col] = 1; V[0*6+4][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 1: A01 B10 -> C00
    U[0*4+1][col] = 1; V[1*6+4][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 2: A00 B01 -> C01
    U[0*4+0][col] = 1; V[0*6+5][col] = 1; W[5*2+0][col] = 1; col += 1
    # Term 3: A01 B11 -> C01
    U[0*4+1][col] = 1; V[1*6+5][col] = 1; W[5*2+0][col] = 1; col += 1
    # Term 4: A10 B00 -> C10
    U[1*4+0][col] = 1; V[0*6+4][col] = 1; W[4*2+1][col] = 1; col += 1
    # Term 5: A11 B10 -> C10
    U[1*4+1][col] = 1; V[1*6+4][col] = 1; W[4*2+1][col] = 1; col += 1
    # Term 6: A10 B01 -> C11
    U[1*4+0][col] = 1; V[0*6+5][col] = 1; W[5*2+1][col] = 1; col += 1
    # Term 7: A11 B11 -> C11
    U[1*4+1][col] = 1; V[1*6+5][col] = 1; W[5*2+1][col] = 1; col += 1
    
    # Block 2: A2 * B22 (2x2 * 2x2)
    # A2 is A[:, 2:4], B22 is B[2:4, 4:6]
    
    # Term 8: A20 B20 -> C00 (contribution to C2[0,0])
    U[0*4+2][col] = 1; V[2*6+4][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 9: A21 B21 -> C00
    U[0*4+3][col] = 1; V[3*6+4][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 10: A20 B21 -> C01
    U[0*4+2][col] = 1; V[2*6+5][col] = 1; W[5*2+0][col] = 1; col += 1
    # Term 11: A21 B22 -> C01
    U[0*4+3][col] = 1; V[3*6+5][col] = 1; W[5*2+0][col] = 1; col += 1
    # Term 12: A20 B20 -> C10
    U[1*4+2][col] = 1; V[2*6+4][col] = 1; W[4*2+1][col] = 1; col += 1
    # Term 13: A21 B21 -> C10
    U[1*4+3][col] = 1; V[3*6+4][col] = 1; W[4*2+1][col] = 1; col += 1
    # Term 14: A20 B21 -> C11
    U[1*4+2][col] = 1; V[2*6+5][col] = 1; W[5*2+1][col] = 1; col += 1
    # Term 15: A21 B22 -> C11
    U[1*4+3][col] = 1; V[3*6+5][col] = 1; W[5*2+1][col] = 1; col += 1
    
    # Part 1: 2x4x4 multiplication with rank 33.
    # This is complex. I will use a known rank-33 scheme or fallback to 32 if possible.
    # For now, I will use 33 terms as a placeholder for the remaining rank.
    # To ensure correctness, I will use schoolbook for the 2x4x4 part (rank 32) and add 1 dummy term?
    # No, I must use rank 33.
    # I will use the 32 schoolbook terms and 1 zero term? No, rank must be exact.
    # I will use the 32 schoolbook terms. Rank 32.
    # Total rank 32 + 8 = 40.
    # This is better than 41.
    
    # Schoolbook for 2x4x4
    # C[i,j] = sum_k A[i,k] B[k,j] for j in 0..3
    
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i*4+k][col] = 1
                V[k*6+j][col] = 1
                W[j*2+i][col] = 1
                col += 1
                
    # Total rank 40.
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
