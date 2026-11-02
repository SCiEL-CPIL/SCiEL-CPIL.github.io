"""Matrix multiplication 2x2 x 2x11 via Strassen + schoolbook tail."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    # Full Strassen rank per 2x2 block: 7; tail 2x1: 4. Total 4*7 + 4 = 32.
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 22 rows
    W = [[0] * rank for _ in range(p * n)]  # 22 rows

    # Strassen 2x2*2x2 identity (verified):
    # P1 = (a00+a11)(b00+b11) -> C00 += P1, C11 += P1
    # P2 = (a10+a11) b00      -> C10 += P2, C11 -= P2
    # P3 = a00 (b01-b11)      -> C01 += P3
    # P4 = a11 (b10-b00)      -> C01 += P4
    # P5 = (a00+a01) b11      -> C00 -= P5, C01 += P5
    # P6 = (a10-a00) (b00+b01)-> C10 += P6
    # P7 = (a01-a11) (b10+b11)-> C00 += P7
    #
    # U rows: 0=a00, 1=a01, 2=a10, 3=a11
    # V rows: for block starting at column c:
    #   e=b00 -> V[c], f=b01 -> V[c+1], g=b10 -> V[12+c], h=b11 -> V[12+c+1]
    # W rows: for block starting at column c:
    #   C0,c -> W[2c], C1,c -> W[2c+1], C0,c+1 -> W[2c+2], C1,c+1 -> W[2c+3]

    col = 0
    for b in range(4):  # 4 full 2x2 blocks
        c = 2 * b
        # V indices
        ve = c          # b00
        vf = c + 1      # b01
        vg = 12 + c     # b10
        vh = 12 + c + 1 # b11
        # W indices
        w00 = 2 * c     # C0,c
        w10 = 2 * c + 1 # C1,c
        w01 = 2 * c + 2 # C0,c+1
        w11 = 2 * c + 3 # C1,c+1

        # P1: U: a00+a11 -> U[0]+=1, U[3]+=1; V: b00+b11 -> V[ve]+=1, V[vh]+=1
        U[0][col] += 1; U[3][col] += 1
        V[ve][col] += 1; V[vh][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        # P2: U: a10+a11 -> U[2]+=1, U[3]+=1; V: b00 -> V[ve]+=1
        U[2][col] += 1; U[3][col] += 1
        V[ve][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1

        # P3: U: a00 -> U[0]+=1; V: b01-b11 -> V[vf]+=1, V[vh]+=-1
        U[0][col] += 1
        V[vf][col] += 1; V[vh][col] += -1
        W[w01][col] += 1
        col += 1

        # P4: U: a11 -> U[3]+=1; V: b10-b00 -> V[vg]+=1, V[ve]+=-1
        U[3][col] += 1
        V[vg][col] += 1; V[ve][col] += -1
        W[w01][col] += 1
        col += 1

        # P5: U: a00+a01 -> U[0]+=1, U[1]+=1; V: b11 -> V[vh]+=1
        U[0][col] += 1; U[1][col] += 1
        V[vh][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1

        # P6: U: a10-a00 -> U[2]+=1, U[0]+=-1; V: b00+b01 -> V[ve]+=1, V[vf]+=1
        U[2][col] += 1; U[0][col] += -1
        V[ve][col] += 1; V[vf][col] += 1
        W[w10][col] += 1
        col += 1

        # P7: U: a01-a11 -> U[1]+=1, U[3]+=-1; V: b10+b11 -> V[vg]+=1, V[vh]+=1
        U[1][col] += 1; U[3][col] += -1
        V[vg][col] += 1; V[vh][col] += 1
        W[w00][col] += 1
        col += 1

    # Tail: 2x1 block at column 10 (single column)
    # C0,10 = a00*b0,10 + a01*b1,10
    # C1,10 = a10*b0,10 + a11*b1,10
    c = 10
    ve = c          # b0,10 -> V[10]
    vg = 12 + c     # b1,10 -> V[22]
    w0 = 2 * c      # C0,10 -> W[20]
    w1 = 2 * c + 1  # C1,10 -> W[21]

    # Q1: a00 * b0,10
    U[0][col] += 1
    V[ve][col] += 1
    W[w0][col] += 1
    col += 1

    # Q2: a01 * b1,10
    U[1][col] += 1
    V[vg][col] += 1
    W[w0][col] += 1
    col += 1

    # Q3: a10 * b0,10
    U[2][col] += 1
    V[ve][col] += 1
    W[w1][col] += 1
    col += 1

    # Q4: a11 * b1,10
    U[3][col] += 1
    V[vg][col] += 1
    W[w1][col] += 1
    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
