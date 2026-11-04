"""Matrix multiplication 6,6,6 via Kronecker product of Strassen(2x2) x schoolbook(3x3)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (6, 6, 6)

    # --- Strassen 2x2 decomposition (rank 7) ---
    # Index: U2[i,j,c] = A_i*A_j, V2[j,k,c] = B_j*B_k, W2[k,i,c] = C_k*C_i
    # Each c is a (left_pair, right_pair) product of two entries.
    # left_pair (i,j) -> (0,1),(1,0),(0,0),(1,1)
    # right_pair (j,k) -> (0,1),(0,1),(1,0),(1,0),(0,0),(1,1),(1,0)
    left_pairs = [(0, 1), (1, 0), (0, 0), (1, 1)]
    right_pairs = [(0, 1), (0, 1), (1, 0), (1, 0), (0, 0), (1, 1), (1, 0)]
    R2 = 7

    U2 = [[0.0] * 2 for _ in range(2)]  # U2[i][j] -> list over c
    V2 = [[0.0] * 2 for _ in range(2)]
    W2 = [[0.0] * 2 for _ in range(2)]
    # build per-c arrays
    U2c = [[0.0] * R2 for _ in range(2 * 2)]   # [i*2+j]
    V2c = [[0.0] * R2 for _ in range(2 * 2)]   # [j*2+k]
    W2c = [[0.0] * R2 for _ in range(2 * 2)]   # [k*2+i]
    for c in range(R2):
        li, lj = left_pairs[c]
        ri, rj = right_pairs[c]
        # U2[li][lj] += 1 ; U2[lj][li] += 1
        U2c[li * 2 + lj][c] += 1
        U2c[lj * 2 + li][c] += 1
        # V2[ri][rj] += 1 ; V2[rj][ri] += 1
        V2c[ri * 2 + rj][c] += 1
        V2c[rj * 2 + ri][c] += 1
        # W2[ri][li] += 1 ; W2[rj][lj] += 1   (W2[k][i], k from right, i from left)
        W2c[ri * 2 + li][c] += 1
        W2c[rj * 2 + lj][c] += 1

    # --- Schoolbook 3x3 decomposition (rank 27) ---
    R3 = 9
    U3c = [[0.0] * R3 for _ in range(3 * 3)]   # U3[i*3+j] = delta(i,j)
    V3c = [[0.0] * R3 for _ in range(3 * 3)]   # V3[j*3+k] = delta(j,k)
    W3c = [[0.0] * R3 for _ in range(3 * 3)]   # W3[k*3+i] = delta(k,i)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c3 = i * 3 + j  # = j*3+k = k*3+i when i=j=k
                U3c[i * 3 + j][c3] += 1.0
                V3c[j * 3 + k][c3] += 1.0
                W3c[k * 3 + i][c3] += 1.0

    # --- Combine: 6x6x6, rank = R2 * R3 = 189 ---
    R = R2 * R3
    U = [[0.0] * R for _ in range(n * m)]  # rows i*6+j, i=(a,b), j=(c,d)
    V = [[0.0] * R for _ in range(m * p)]  # rows j*6+k, j=(c,d), k=(e,f)
    W = [[0.0] * R for _ in range(p * n)]  # rows k*6+i, k=(e,f), i=(a,b)

    # map 3-block index b in {0,1,2}
    for a in range(2):
        for b in range(3):
            for c in range(2):
                for d in range(3):
                    # U row: i=(a,b) -> a*3+b ; j=(c,d) -> c*3+d
                    i6 = a * 3 + b
                    j6 = c * 3 + d
                    u2 = U2c[a * 2 + c]
                    u3 = U3c[b * 3 + d]
                    for alpha in range(R2):
                        ua = u2[alpha]
                        if ua == 0.0:
                            continue
                        for gamma in range(R3):
                            ug = u3[gamma]
                            if ug == 0.0:
                                continue
                            U[i6 * 6 + j6][alpha * R3 + gamma] += ua * ug

    for c in range(2):
        for d in range(3):
            for e in range(2):
                for f in range(3):
                    # V row: j=(c,d) -> c*3+d ; k=(e,f) -> e*3+f
                    j6 = c * 3 + d
                    k6 = e * 3 + f
                    v2 = V2c[c * 2 + e]
                    v3 = V3c[d * 3 + f]
                    for alpha in range(R2):
                        va = v2[alpha]
                        if va == 0.0:
                            continue
                        for gamma in range(R3):
                            vg = v3[gamma]
                            if vg == 0.0:
                                continue
                            V[j6 * 6 + k6][alpha * R3 + gamma] += va * vg

    for e in range(2):
        for f in range(3):
            for a in range(2):
                for b in range(3):
                    # W row: k=(e,f) -> e*3+f ; i=(a,b) -> a*3+b
                    k6 = e * 3 + f
                    i6 = a * 3 + b
                    w2 = W2c[e * 2 + a]
                    w3 = W3c[f * 3 + b]
                    for alpha in range(R2):
                        wa = w2[alpha]
                        if wa == 0.0:
                            continue
                        for gamma in range(R3):
                            wg = w3[gamma]
                            if wg == 0.0:
                                continue
                            W[k6 * 6 + i6][alpha * R3 + gamma] += wa * wg

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
