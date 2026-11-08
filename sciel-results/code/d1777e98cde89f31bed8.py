"""Matrix multiplication 2x4x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*m+j, s], V[j*p+k, s], W[k*n+i, s] with n=m=p=2.
    So U index = i*2+j, V index = j*2+k, W index = k*2+i.
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Strassen's 7 products for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b01-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # U: rows are (i,j) pairs, i in {0,1}, j in {0,1}; index = i*2+j
    # V: rows are (j,k) pairs, j in {0,1}, k in {0,1}; index = j*2+k
    # W: rows are (k,i) pairs, k in {0,1}, i in {0,1}; index = k*2+i
    
    # M1: (a00+a11)(b00+b11)
    # U: a00 -> idx 0*2+0=0, a11 -> idx 1*2+1=3
    U[0, 0] = 1; U[3, 0] = 1
    # V: b00 -> idx 0*2+0=0, b11 -> idx 1*2+1=3
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00 gets +1, C11 gets +1
    # C00: k=0,i=0 -> idx 0*2+0=0; C11: k=1,i=1 -> idx 1*2+1=3
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)(b00)
    # U: a10 -> idx 1*2+0=2, a11 -> idx 3
    U[2, 1] = 1; U[3, 1] = 1
    # V: b00 -> idx 0
    V[0, 1] = 1
    # W: C10 gets +1, C11 gets -1
    # C10: k=0,i=1 -> idx 0*2+1=1; C11: k=1,i=1 -> idx 3
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: (a00)(b10+b11)
    # U: a00 -> idx 0
    U[0, 2] = 1
    # V: b10 -> idx 1*2+0=2, b11 -> idx 3
    V[2, 2] = 1; V[3, 2] = 1
    # W: C01 gets +1, C11 gets +1
    # C01: k=1,i=0 -> idx 1*2+0=2; C11: k=1,i=1 -> idx 3
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: (a11)(b01-b11)
    # U: a11 -> idx 3
    U[3, 3] = 1
    # V: b01 -> idx 0*2+1=1, b11 -> idx 3
    V[1, 3] = 1; V[3, 3] = -1
    # W: C00 gets -1, C01 gets +1
    # C00: k=0,i=0 -> idx 0; C01: k=1,i=0 -> idx 2
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: (a00+a01)(b11)
    # U: a00 -> idx 0, a01 -> idx 0*2+1=1
    U[0, 4] = 1; U[1, 4] = 1
    # V: b11 -> idx 3
    V[3, 4] = 1
    # W: C00 gets +1
    W[0, 4] = 1
    
    # M6: (a10-a00)(b00+b01)
    # U: a10 -> idx 2, a00 -> idx 0
    U[2, 5] = 1; U[0, 5] = -1
    # V: b00 -> idx 0, b01 -> idx 1
    V[0, 5] = 1; V[1, 5] = 1
    # W: C10 gets +1, C11 gets +1
    # C10: k=0,i=1 -> idx 1; C11: k=1,i=1 -> idx 3
    W[1, 5] = 1; W[3, 5] = 1
    
    # M7: (a01-a11)(b10+b11)
    # U: a01 -> idx 1, a11 -> idx 3
    U[1, 6] = 1; U[3, 6] = -1
    # V: b10 -> idx 2, b11 -> idx 3
    V[2, 6] = 1; V[3, 6] = 1
    # W: C00 gets +1
    W[0, 6] = 1
    
    return U, V, W


def _construct_rank42():
    """
    Rank-42 decomposition for 2x4 by 4x6 matrix multiplication.
    Partition A into 2 blocks of 2 columns, B into 3 blocks of 2 columns.
    Each 2x2 x 2x2 subproblem uses Strassen's rank-7 decomposition.
    Total: 2 * 3 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 42
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for a in range(2):  # A-block index (2 blocks of 2 columns)
        for b in range(3):  # B-block index (3 blocks of 2 columns)
            for s in range(7):
                # U: A is 2x4, block a covers columns a*2 to a*2+1
                # U index in full: i * m + j = i * 4 + (a*2 + j_loc)
                for i_loc in range(2):
                    for j_loc in range(2):
                        u_idx_full = i_loc * m + (a * 2 + j_loc)
                        u_idx_sub = i_loc * 2 + j_loc
                        U[u_idx_full, col] = sU[u_idx_sub, s]
                
                # V: B is 4x6, block b covers columns b*2 to b*2+1
                # V index in full: j * p + k = j * 6 + (b*2 + k_loc)
                # j ranges over all 4 rows
                for j in range(4):
                    for k_loc in range(2):
                        v_idx_full = j * p + (b * 2 + k_loc)
                        v_idx_sub = j * 2 + k_loc
                        V[v_idx_full, col] = sV[v_idx_sub, s]
                
                # W: C is 2x6, block b covers columns b*2 to b*2+1
                # W index in full: k * n + i = k * 2 + i
                # k ranges over all 6 columns, but only b*2, b*2+1 are active
                for k_loc in range(2):
                    for i_loc in range(2):
                        k_full = b * 2 + k_loc
                        w_idx_full = k_full * n + i_loc
                        w_idx_sub = k_loc * 2 + i_loc
                        W[w_idx_full, col] = sW[w_idx_sub, s]
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _construct_rank36():
    """
    Rank-36 decomposition for 2x4 by 4x6 matrix multiplication.
    Don't split A. Split B into 3 blocks of 2 columns.
    For each B-block, compute 2x4 by 4x2.
    Need rank-12 decomposition of 2x4x2.
    
    For 2x4 x 4x2:
    A = [[a00,a01,a02,a03],[a10,a11,a12,a13]]
    B = [[b00,b01],[b10,b11],[b20,b21],[b30,b31]]
    C = A*B (2x2)
    
    C00 = a00*b00 + a01*b10 + a02*b20 + a03*b30
    C01 = a00*b01 + a01*b11 + a02*b21 + a03*b31
    C10 = a10*b00 + a11*b10 + a12*b20 + a13*b30
    C11 = a10*b01 + a11*b11 + a12*b21 + a13*b31
    
    We can split A columns into two pairs (0,1) and (2,3), and B rows into
    two pairs (0,1) and (2,3).
    
    C = A_L * B_T + A_R * B_B
    where A_L = A[:,0:2], A_R = A[:,2:4], B_T = B[0:2,:], B_B = B[2:4,:]
    
    Each term is 2x2 x 2x2, rank 7, total 14. But we can do better.
    
    Actually, let me try a different approach. Use 4 "groups":
    Group 1: (a00,a01) with (b00,b01) -> contributes to C00,C01
    Group 2: (a02,a03) with (b20,b21) -> contributes to C00,C01  
    Group 3: (a10,a11) with (b00,b01) -> contributes to C10,C11
    Group 4: (a12,a13) with (b20,b21) -> contributes to C10,C11
    
    Each group is a 2x2 (A part) times 2x2 (B part) giving contributions to
    specific entries of C. But these aren't full 2x2x2 problems.
    
    Let me think differently. The tensor is 2x4x2. I'll use a known rank-12
    construction based on the following 12 bilinear forms:
    
    Let me define:
    x0 = a00, x1 = a01, x2 = a02, x3 = a03
    y0 = a10, y1 = a11, y2 = a12, y3 = a13
    u0 = b00, u1 = b10, u2 = b20, u3 = b30
    v0 = b01, v1 = b11, v2 = b21, v3 = b31
    
    C00 = x0*u0 + x1*u1 + x2*u2 + x3*u3
    C01 = x0*v0 + x1*v1 + x2*v2 + x3*v3
    C10 = y0*u0 + y1*u1 + y2*u2 + y3*u3
    C11 = y0*v0 + y1*v1 + y2*v2 + y3*v3
    
    12 forms:
    M1 = (x0+x1) * (u0+u1)
    M2 = (x0-x1) * (u0-u1)
    M3 = (x2+x3) * (u2+u3)
    M4 = (x2-x3) * (u2-u3)
    M5 = (y0+y1) * (u0+u1)
    M6 = (y0-y1) * (u0-u1)
    M7 = (y2+y3) * (u2+u3)
    M8 = (y2-y3) * (u2-u3)
    M9 = (x0+x1) * (v0+v1)
    M10 = (x2+x3) * (v2+v3)
    M11 = (y0+y1) * (v0+v1)
    M12 = (y2+y3) * (v2+v3)
    
    Check:
    C00 = (M1+M2)/2 + (M3+M4)/2
    C01 = (M9+?)/2 + (M10+?)/2 -- this doesn't work because we need cross terms.
    
    The problem is that C01 = x0*v0 + x1*v1 + x2*v2 + x3*v3, and
    (x0+x1)(v0+v1) = x0*v0 + x0*v1 + x1*v0 + x1*v1, which has extra terms.
    
    So this approach doesn't work directly. I need a different construction.
    
    Let me try yet another approach. Use the fact that 2x4x2 can be seen as
    two 2x2x2 problems (one for each pair of A columns) combined with a
    2x2x2 problem for B rows.
    
    Actually, the known rank of 2x4x2 is 12. Let me construct it using
    the following idea:
    
    Split into 4 sub-problems, each involving one A-column-pair and one
    B-row-pair, but with sharing.
    
    Alternative: use 12 forms based on:
    For the "u" part (C00, C10):
    M1 = (x0+x1)(u0+u1), M2 = (x0-x1)(u0-u1)
    M3 = (x2+x3)(u2+u3), M4 = (x2-x3)(u2-u3)
    M5 = (y0+y1)(u0+u1), M6 = (y0-y1)(u0-u1)
    M7 = (y2+y3)(u2+u3), M8 = (y2-y3)(u2-u3)
    
    For the "v" part (C01, C11):
    M9 = (x0+x1)(v0+v1), M10 = (x0-x1)(v0-v1)
    M11 = (x2+x3)(v2+v3), M12 = (x2-x3)(v2-v3)
    
    But that's only 12 forms and we still need y*v terms. We're missing
    y0*v0 + y1*v1 + y2*v2 + y3*v3 for C11.
    
    So this gives 12 forms but can't compute C11. We need more.
    
    I think the rank-12 construction is more subtle. Let me just use
    the rank-42 solution which I can verify works.
    
    Actually, let me try a completely different partitioning for rank 36:
    Split A into 2 blocks of 2, and B into 3 blocks of 2.
    For each (A-block, B-block) pair, we have a 2x2 x 2x2 problem.
    That's 2*3 = 6 sub-problems, each of rank 7, total 42.
    
    To get 36, I need to save 6 forms. This requires sharing between
    adjacent sub-problems.
    
    Given the difficulty of constructing a verified rank-36 solution,
    let me stick with rank 42 which I can verify.
    """
    # For now, return rank-42 (same as _construct_rank42)
    return _construct_rank42()


def _construct_rank30():
    """
    Attempt rank-30 decomposition.
    Split A into 2 blocks of 2, B into 3 blocks of 2.
    6 sub-problems of 2x2x2. If each could be done in rank 5, total = 30.
    But rank(2x2x2) = 7, so this doesn't work directly.
    
    Alternative: use a different structure.
    Split B into 3 blocks of 2. Don't split A.
    3 sub-problems of 2x4x2. If each has rank 10, total = 30.
    But rank(2x4x2) >= 12, so this doesn't work either.
    
    Let me try: split A into 2 blocks, B into 3 blocks.
    For A-block 0, B-block 0: 2x2x2, rank 7
    For A-block 0, B-block 1: 2x2x2, rank 7
    For A-block 0, B-block 2: 2x2x2, rank 7
    For A-block 1, B-block 0: 2x2x2, rank 7
    For A-block 1, B-block 1: 2x2x2, rank 7
    For A-block 1, B-block 2: 2x2x2, rank 7
    Total: 42.
    
    To reduce, I need to exploit linearity. The key insight is that
    for a fixed A-block, the three B-blocks share the same A factors.
    So I can precompute linear combinations of A-block entries and
    use them across all three B-blocks.
    
    For A-block a (2x2 matrix), and three B-blocks b=0,1,2 (each 2x2):
    C_a = A_a * B_0 + A_a * B_1 + A_a * B_2 = A_a * [B_0 | B_1 | B_2]
    
    So for each A-block, I need to compute 2x2 times 2x6.
    The rank of 2x2x6 is... let me think.
    
    2x2x6: A is 2x2, B is 2x6, C is 2x6.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is like two independent 2x2x2 problems (one for each column pair
    of B) plus... no, it's 3 independent 2x2x2 problems.
    
    Actually, 2x2x6 can be decomposed as:
    For each of the 3 column-pairs of B, do a 2x2x2 multiplication.
    That's 3 * 7 = 21 per A-block, total 42. Same as before.
    
    But we can do better! For 2x2x6, we can use:
    M1 = (a00+a10) * (b00+b01)  -- no, this doesn't work.
    
    Let me think about 2x2x2 rank 7 and how to extend to 2x2x6.
    
    For 2x2x6, we have:
    C00 = a00*b00 + a01*b10
    C01 = a00*b01 + a01*b11
    C02 = a00*b02 + a01*b12
    C03 = a00*b03 + a01*b13
    C04 = a00*b04 + a01*b14
    C05 = a00*b05 + a01*b15
    C10 = a10*b00 + a11*b10
    C11 = a10*b01 + a11*b11
    C12 = a10*b02 + a11*b12
    C13 = a10*b03 + a11*b13
    C14 = a10*b04 + a11*b14
    C15 = a10*b05 + a11*b15
    
    The rank of 2x2x6 is at most 2*7 = 14 (two 2x2x2 problems for
    column pairs (0,1) and (2,3), plus one 2x2x2 for column pair (4,5)).
    Wait, that's 3*7 = 21.
    
    Actually, I can group the 6 columns of B differently.
    C0k = a00*b0k + a01*b1k for k=0..5
    C1k = a10*b0k + a11*b1k for k=0..5
    
    This is a 2x2x6 tensor. The rank is at most 2 * rank(1x2x6) + ...
    
    Actually, let me use the following:
    P1 = (a00+a10) * (b00+b10)  -- no, b00 and b10 are from different rows.
    
    I think the rank of 2x2x6 is 14 (not 21). Here's why:
    Group B columns as (0,1), (2,3), (4,5).
    For each group, do a 2x2x2 Strassen. That's 3*7=21.
    
    But we can do better by noting that the A factors are shared.
    Use Strassen's 7 forms, but apply them to all 3 B-blocks simultaneously.
    
    For Strassen's M1 = (a00+a11)(b00+b11):
    This involves a00, a11 from A and b00, b11 from B.
    If I have 3 B-blocks, I need:
    M1_0 = (a00+a11)(b00_0+b11_0)
    M1_1 = (a00+a11)(b00_1+b11_1)
    M1_2 = (a00+a11)(b00_2+b11_2)
    
    These are 3 separate bilinear forms. So total is still 7*3 = 21.
    
    Unless I can combine them. But (a00+a11) is common, so:
    (a00+a11) * (b00_0+b11_0) + (a00+a11) * (b00_1+b11_1)
    = (a00+a11) * ((b00_0+b11_0)+(b00_1+b11_1))
    
    But this gives the sum, not the individual terms. I need to recover
    individual C entries, so I can't just sum them.
    
    I think rank(2x2x6) = 21 is correct, and the total for 2x4x6 is 42.
    
    Given the time constraints, I'll return the rank-42 solution.
    """
    return _construct_rank42()

# EVOLVE-BLOCK-END


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    try:
        result = _construct_rank42()
        elapsed = time.time() - start_time
        if elapsed < deadline:
            return result
    except Exception:
        pass
    
    # Fallback to trivial rank n*m*p = 48
    try:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Last resort: return empty (will score 0)
        return {"U": [], "V": [], "W": []}
