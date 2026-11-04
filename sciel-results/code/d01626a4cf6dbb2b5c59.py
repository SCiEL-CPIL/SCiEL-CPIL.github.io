"""Matrix multiplication 4x4x8 via two rank-49 4x4x4 Strassen blocks (rank 98)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def _build_4x4x4_rank49():
    n = 4
    # Outer Strassen (block level, 2x2 blocks of 2x2)
    A_out = {
        1: [((0,0),1), ((1,1),1)],
        2: [((1,0),1), ((1,1),1)],
        3: [((0,0),1)],
        4: [((1,1),1)],
        5: [((0,0),1), ((0,1),1)],
        6: [((1,0),1), ((0,0),-1)],
        7: [((0,1),1), ((1,1),-1)],
    }
    B_out = {
        1: [((0,0),1), ((1,1),1)],
        2: [((0,0),1)],
        3: [((0,1),1), ((1,1),-1)],
        4: [((1,0),1), ((0,0),-1)],
        5: [((1,1),1)],
        6: [((0,0),1), ((0,1),1)],
        7: [((1,0),1), ((1,1),1)],
    }
    C_out = {
        (0,0): {1:1, 4:1, 5:-1, 7:1},
        (0,1): {3:1, 5:1},
        (1,0): {2:1, 6:1},
        (1,1): {1:1, 2:-1, 3:1, 6:1},
    }
    # Inner Strassen (scalar 2x2x2)
    A_in = {
        1: [((0,0),1), ((1,1),1)],
        2: [((1,0),1), ((1,1),1)],
        3: [((0,0),1)],
        4: [((1,1),1)],
        5: [((0,0),1), ((0,1),1)],
        6: [((1,0),1), ((0,0),-1)],
        7: [((0,1),1), ((1,1),-1)],
    }
    B_in = {
        1: [((0,0),1), ((1,1),1)],
        2: [((0,0),1)],
        3: [((0,1),1), ((1,1),-1)],
        4: [((1,0),1), ((0,0),-1)],
        5: [((1,1),1)],
        6: [((0,0),1), ((0,1),1)],
        7: [((1,0),1), ((1,1),1)],
    }
    C_in = {
        (0,0): {1:1, 4:1, 5:-1, 7:1},
        (0,1): {3:1, 5:1},
        (1,0): {2:1, 6:1},
        (1,1): {1:1, 2:-1, 3:1, 6:1},
    }
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    col = 0
    for o in range(1, 8):
        for s in range(1, 8):
            for (i1, j1), ca in A_out[o]:
                for (i0, j0), da in A_in[s]:
                    U[(2*i1+i0)*n + (2*j1+j0)][col] += ca*da
            for (j1, k1), cb in B_out[o]:
                for (j0, k0), db in B_in[s]:
                    V[(2*j1+j0)*n + (2*k1+k0)][col] += cb*db
            for (k1, i1), cdict in C_out.items():
                cc = cdict.get(o, 0)
                if cc:
                    for (k0, i0), didict in C_in.items():
                        dc = didict.get(s, 0)
                        if dc:
                            W[(2*k1+k0)*n + (2*i1+i0)][col] += cc*dc
            col += 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s
    n, m, p = DIMENSIONS  # 4, 4, 8
    U49, V49, W49 = _build_4x4x4_rank49()
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U49[i][c]
            U[i][49+c] = U49[i][c]
    for j in range(m):
        for k in range(4):
            for c in range(49):
                V[j*p + k][c] = V49[j*n + k][c]
                V[j*p + (k+4)][49+c] = V49[j*n + k][c]
    for k in range(4):
        for i in range(n):
            for c in range(49):
                W[k*n + i][c] = W49[k*n + i][c]
                W[(k+4)*n + i][49+c] = W49[k*n + i][c]
    # Verify (cheap): T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == (i==k)
    if time.time() < deadline:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != (1 if i == k else 0):
                        raise ValueError("decomposition failed at %d,%d,%d got %r" % (i, j, k, s))
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
