import time

def construct():
    t0 = time.time()
    # 2x4 * 4x8  ->  2x8
    # Use 8 Kronecker blocks, each rank 7  =>  rank 56
    n = 2
    m = 4
    p = 8

    R = 56
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]

    # Local rank-7 Strassen template for (2x2)*(2x2)
    # Indices 0..3 correspond to local rows/cols 0,1,2,3 = (0,0),(0,1),(1,0),(1,1)
    # r=0: (a00+a11)(b00+b11) -> c00+1, c11+1
    U0 = [0,0,0,0]
    U1 = [0,0,0,0]
    U2 = [0,0,0,0]
    U3 = [0,0,0,0]
    U4 = [0,0,0,0]
    U5 = [0,0,0,0]
    U6 = [0,0,0,0]
    V0 = [0,0,0,0]
    V1 = [0,0,0,0]
    V2 = [0,0,0,0]
    V3 = [0,0,0,0]
    V4 = [0,0,0,0]
    V5 = [0,0,0,0]
    V6 = [0,0,0,0]
    W0 = [0,0,0,0]
    W1 = [0,0,0,0]
    W2 = [0,0,0,0]
    W3 = [0,0,0,0]

    U0[0] = 1; U0[3] = 1
    V0[0] = 1; V0[3] = 1
    W0[0] = 1; W0[3] = 1

    U1[2] = 1; U1[3] = 1
    V1[0] = 1
    W1[2] = 1; W1[3] = -1

    U2[0] = 1
    V2[1] = 1; V2[3] = -1
    W2[1] = 1; W2[3] = 1

    U3[3] = 1
    V3[2] = 1; V3[0] = -1
    W3[0] = 1; W3[2] = 1

    U4[0] = 1; U4[1] = 1
    V4[3] = 1
    W4[0] = -1; W4[1] = 1

    U5[2] = 1; U5[0] = -1
    V5[0] = 1; V5[1] = 1
    W5[3] = 1

    U6[1] = 1; U6[3] = -1
    V6[2] = 1; V6[3] = 1
    W6[0] = 1

    UU = (U0,U1,U2,U3,U4,U5,U6)
    VV = (V0,V1,V2,V3,V4,V5,V6)
    WW = (W0,W1,W2,W3,W4,W5,W6)

    # Place 8 blocks: 2 over the middle (m) dimension, 4 over the output (p) dimension
    group = 0
    for half in (0, 1):
        for jj in range(4):
            base = group * 7
            group += 1

            # A rows: (row, local_col)
            a_idx = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            b_idx = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            c_idx = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for k in range(7):
                rr = base + k
                u_row = UU[k]
                v_row = VV[k]
                w_row = WW[k]
                for s in range(4):
                    if u_row[s]:
                        U[a_idx[s]][rr] += u_row[s]
                    if v_row[s]:
                        V[b_idx[s]][rr] += v_row[s]
                    if w_row[s]:
                        W[c_idx[s]][rr] += w_row[s]

    if time.time() - t0 > 115:
        pass

    return {"U": U, "V": V, "W": W}
