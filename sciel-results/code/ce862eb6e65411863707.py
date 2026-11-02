import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Direct structural decomposition via 2x2 Strassen tiling.
    # Output C (n x p) is split into (n x 2) column-pair blocks.
    # For each column-pair jj (0..p/2-1) and a row-pair selector s (0..n/2-1),
    # we solve a 2x2 matrix multiply using 7 rank-1 tensors (Strassen).
    # Total rank = (p/2) * (n/2) * 7 = 4 * 1 * 7 = 28? No: n/2=1, so (p/2)*(n/2)*7=4*1*7=28.
    #
    # Re-examine: A is n x m = 2 x 4. Split A along its m dimension into (m/2)=2 halves,
    # each n x 2 = 2 x 2. Split B (m x p = 4 x 8) along m into 2 halves each 2 x 8,
    # and along p into (p/2)=4 blocks each 2 x 2. So the (2x2) sub-products:
    # C[:, 2jj:2jj+2] = A[:,0:2] @ B[0:2, 2jj:2jj+2] + A[:,2:4] @ B[2:4, 2jj:2jj+2]
    # = 2 terms, each a 2x2 x 2x2 -> 2x2 multiply (rank 7). So per jj: 2 * 7 = 14,
    # over 4 jj blocks -> 56. Confirmed.

    # Strassen 2x2 local factorization (rank 7).
    # Local A indices a00=0,a01=1,a10=2,a11=3 ; local B b00=0,b01=1,b10=2,b11=3
    # local C c00=0,c01=1,c10=2,c11=3
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

    # Global index helpers.
    # A global index = i*n? No, A is n x m, flat row-major i*m+j.
    # B global index = j_g * p + k_g.
    # W convention: W[k*n + i, r]  -> output index = k * n + i.

    group = 0
    for jj in range(p // 2):        # column-pair of output
        for half in range(2):       # which A-half / B-half (0: m[0:2], 1: m[2:4])
            base_r = group * 7
            group += 1

            # A-half 'half': columns 2*half, 2*half+1
            # local a00->(i0, j=2h), a01->(i0, j=2h+1), a10->(i1, j=2h), a11->(i1, j=2h+1)
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            # B-half 'half': rows 2*half, 2*half+1 ; cols 2*jj, 2*jj+1
            # local b00->(row=2h, col=2jj), b01->(row=2h, col=2jj+1), b10->(row=2h+1, col=2jj), b11->(row=2h+1, col=2jj+1)
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

    # Safety deadline guard: deterministic build, well under budget.
    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
