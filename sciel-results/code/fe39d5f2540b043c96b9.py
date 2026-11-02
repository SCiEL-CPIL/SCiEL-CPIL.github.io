import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-12 decomposition for 2x4 * 4x2 -> 2x2
    # A: 2x4, indices: a[i][j] -> flat i*4+j, i in {0,1}, j in {0,1,2,3}
    # B: 4x2, indices: b[j][k] -> flat j*2+k, j in {0,1,2,3}, k in {0,1}
    # C: 2x2, indices: c[i][k] -> W index k*n+i = k*2+i, i in {0,1}, k in {0,1}
    #
    # We decompose using 12 rank-1 terms.
    # Strategy: use Strassen on the "outer" 2x2 structure where we treat
    # columns of A and rows of B in pairs, but with shared intermediates.
    #
    # Let's define:
    # A0 = [a00, a01], A1 = [a02, a03] (row 0, col pairs)
    # A2 = [a10, a11], A3 = [a12, a13] (row 1, col pairs)
    # B0 = [b00, b01] (row 0), B1 = [b10, b11] (row 1)
    # B2 = [b20, b21] (row 2), B3 = [b30, b31] (row 3)
    #
    # C = A0*B0 + A1*B2 + A2*B1 + A3*B3  (each is 1x2 * 2x2 -> 1x2... no)
    # Actually: c[i][k] = sum_j a[i][j]*b[j][k]
    # c[0][0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # c[0][1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    # c[1][0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
    # c[1][1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
    #
    # Rank-12 construction:
    # We use the following 12 bilinear forms:
    #
    # Group 1 (for "even" B rows 0,2): 6 terms
    # Group 2 (for "odd" B rows 1,3): 6 terms
    #
    # Actually, let me use a direct rank-12 construction.
    # 
    # M1  = (a00+a01+a02+a03) * (b00+b10+b20+b30)  -> contributes to all c[i][k] with coefficient 1
    # But that's not quite right. Let me use a known construction.
    #
    # I'll use the following approach: split into two 2x2*2x2 multiplications
    # but share 2 of the 7 Strassen factors.
    #
    # Alternative: use rank 12 directly.
    #
    # Let me define 12 rank-1 tensors explicitly.
    # U: 8 x 12, V: 8 x 12, W: 4 x 12
    #
    # A flat indices: 0=a00, 1=a01, 2=a02, 3=a03, 4=a10, 5=a11, 6=a12, 7=a13
    # B flat indices: 0=b00, 1=b01, 2=b10, 3=b11, 4=b20, 5=b21, 6=b30, 7=b31
    # W flat indices: 0=c00, 1=c01, 2=c10, 3=c11 (k*n+i: k=0,i=0->0; k=1,i=0->2; k=0,i=1->1; k=1,i=1->3)
    # Wait: W[k*n+i]. n=2. So:
    #   c[0][0]: k=0,i=0 -> 0*2+0 = 0
    #   c[0][1]: k=1,i=0 -> 1*2+0 = 2
    #   c[1][0]: k=0,i=1 -> 0*2+1 = 1
    #   c[1][1]: k=1,i=1 -> 1*2+1 = 3
    #
    # So W indices: 0=c00, 1=c10, 2=c01, 3=c11

    U_loc = [[0]*12 for _ in range(8)]
    V_loc = [[0]*12 for _ in range(8)]
    W_loc = [[0]*12 for _ in range(4)]

    # r=0: M = (a00+a01+a02+a03) * (b00+b10+b20+b30)
    # This contributes +1 to c00, +1 to c10 (since it's row-sum of A times col-sum of B for col 0)
    # Actually no. Let me be more careful.
    #
    # I'll use a verified rank-12 construction for 2x4*4x2.
    #
    # After research, the rank of the 2x4*4x2 multiplication tensor is 12.
    # Here's a construction:
    #
    # Define:
    # s0 = a00 + a02,  s1 = a01 + a03,  s2 = a10 + a12,  s3 = a11 + a13
    # t0 = b00 + b20,  t1 = b01 + b21,  t2 = b10 + b30,  t3 = b11 + b31
    #
    # And:
    # u0 = a00 - a02,  u1 = a01 - a03,  u2 = a10 - a12,  u3 = a11 - a13
    # v0 = b00 - b20,  v1 = b01 - b21,  v2 = b10 - b30,  v3 = b11 - b31
    #
    # C = (s0*t0 + u0*v0)/2 ... this gets complicated.
    #
    # Let me just use two independent Strassen decompositions (rank 14) 
    # but that gives 56 total. To improve, I need rank 12 per block.
    #
    # Actually, let me just go with the clean rank-56 solution but verify it's correct.
    # The inspirations show the same approach. Let me ensure correctness.

    # Revert to the known-correct Strassen tiling approach (rank 56)
    # but write it cleanly.

    # Strassen 2x2 local factorization (rank 7).
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
        for half in range(2):       # which A-half / B-half
            base_r = group * 7
            group += 1

            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
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

    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
