import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Structural change: instead of 8 independent 2x2 Strassen blocks (rank 56),
    # we exploit the fact that A is 2x4 and B is 4x8. We can view this as:
    # C (2x8) = A (2x4) @ B (4x8)
    # 
    # Key insight: We can use a "block Strassen" approach where we treat the 
    # 4x4 middle dimension more cleverly. Specifically, we can use the fact that
    # multiplying a 2x4 by 4x8 can be done by:
    # 1. Splitting B into two 4x4 blocks: B1 (cols 0-3) and B2 (cols 4-7)
    # 2. For each 4x4 block, we need to compute A @ B_block (2x4 @ 4x4 -> 2x4)
    # 
    # For the 2x4 @ 4x4 -> 2x4 problem, we can use a rank-12 decomposition:
    # A (2x4) = [A0 A1] where A0, A1 are 2x2
    # B (4x4) = [B0; B1] where B0, B1 are 2x4
    # C = A0 @ B0 + A1 @ B1
    # Each A0@B0 is 2x2 @ 2x4 -> 2x4, which can be done with rank 6 (since 
    # it's just 2 scalar multiplications per output element, but we can share).
    #
    # Actually, let me think about this differently. The standard rank for 
    # <2,4,8> is at most 56 (from 8 Strassen blocks). But we can do better by
    # using a different tiling.
    #
    # New approach: Use the "ladder" method or direct rank-1 constructions.
    # The tensor T has dimensions 2*4=8 (A), 4*8=32 (B), 2*8=16 (C).
    # We want to express T as sum of rank-1 tensors.
    #
    # Let me try a rank-48 construction:
    # Split the 4x8 B matrix into 2x4 blocks. We have 2 row-halves and 4 col-pairs.
    # For each of the 2 row-halves of B and each of the 4 col-pairs, we have a 2x4 sub-block.
    # A is 2x4, so we can split A into two 2x2 halves.
    # 
    # Actually, let me just implement the rank-56 Strassen solution correctly first,
    # then try to optimize. The current code has a bug in the U_loc initialization.
    
    # Correct Strassen 2x2 local factorization (rank 7).
    # Local A: a00=0, a01=1, a10=2, a11=3
    # Local B: b00=0, b01=1, b10=2, b11=3
    # Local C: c00=0, c01=1, c10=2, c11=3
    
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]
    
    # r=0: M1=(a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    # r=1: M2=(a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    # r=2: M3=a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    # r=3: M4=a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    # r=4: M5=(a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    # r=5: M6=(a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    # r=6: M7=(a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1
    
    num_blocks = (p // 2) * 2  # 4 column-pair blocks * 2 A-halves = 8 groups
    total_rank = num_blocks * 7  # 56
    
    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56
    
    group = 0
    for jj in range(p // 2):        # column-pair of output
        for half in range(2):       # which A-half / B-half (0: m[0:2], 1: m[2:4])
            base_r = group * 7
            group += 1
            
            # A-half 'half': columns 2*half, 2*half+1
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            # B-half 'half': rows 2*half, 2*half+1 ; cols 2*jj, 2*jj+1
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            # C local: c00->(i0, k=2jj), c01->(i0, k=2jj+1), c10->(i1, k=2jj), c11->(i1, k=2jj+1)
            # W index = k*n + i
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]
            
            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val
    
    if time.time() - start_time > 110:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
