import time
import sys

N, M, P = 2, 4, 7
# Output ordering: W[k*n + i, r]  => W index = k*N + i
# U index = i*M + j   (A[i,j])
# V index = j*P + k   (B[j,k])

def construct():
    t0 = time.time()
    DEADLINE = t0 + 100  # safety margin under 120s

    # Build explicit (Strassen / schoolbook) decomposition
    # We process each j-pair (j0, j0+1) with j0 in {0,2}.
    # For each of the 7 output columns k, we try:
    #   - if k and k+1 both exist, use Strassen 2x2x2 (7 mults)
    #   - else use schoolbook 2x2x1 (4 mults)
    # Total rank per j-pair: 7*3 + 4 = 25  ->  total 50

    U = [[0]*50 for _ in range(N*M)]
    V = [[0]*50 for _ in range(M*P)]
    W = [[0]*50 for _ in range(P*N)]

    col = 0
    for j0 in (0, 2):
        for k0 in (0, 2, 4):
            # Strassen 2x2x2 for A[0:2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[0:2, k0:k0+2]
            a_idx = 0*M + j0
            b_idx = 0*M + (j0+1)
            c_idx = 1*M + j0
            d_idx = 1*M + (j0+1)

            e_idx = j0*P + k0
            f_idx = j0*P + (k0+1)
            g_idx = (j0+1)*P + k0
            h_idx = (j0+1)*P + (k0+1)

            c00 = k0*N + 0
            c01 = (k0+1)*N + 0
            c10 = k0*N + 1
            c11 = (k0+1)*N + 1

            # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
            U[a_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1;  W[c11][col] += 1
            col += 1

            # M2 = (c+d)e -> C10 += M2, C11 -= M2
            U[c_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10][col] += 1;  W[c11][col] -= 1
            col += 1

            # M3 = a(f-h) -> C01 += M3, C11 += M3
            U[a_idx][col] += 1
            V[f_idx][col] += 1; V[h_idx][col] -= 1
            W[c01][col] += 1;  W[c11][col] += 1
            col += 1

            # M4 = d(g-e) -> C00 += M4, C10 += M4
            U[d_idx][col] += 1
            V[g_idx][col] += 1; V[e_idx][col] -= 1
            W[c00][col] += 1;  W[c10][col] += 1
            col += 1

            # M5 = (a+b)h -> C00 -= M5, C01 += M5
            U[a_idx][col] += 1; U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00][col] -= 1;  W[c01][col] += 1
            col += 1

            # M6 = (c-a)(e+f) -> C11 += M6
            U[c_idx][col] += 1; U[a_idx][col] -= 1
            V[e_idx][col] += 1; V[f_idx][col] += 1
            W[c11][col] += 1
            col += 1

            # M7 = (b-d)(g+h) -> C00 += M7
            U[b_idx][col] += 1; U[d_idx][col] -= 1
            V[g_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1
            col += 1

        # schoolbook 2x2x1 for last column k=6
        a_idx = 0*M + j0
        b_idx = 0*M + (j0+1)
        c_idx = 1*M + j0
        d_idx = 1*M + (j0+1)

        e_idx = j0*P + 6
        g_idx = (j0+1)*P + 6

        c00 = 6*N + 0
        c10 = 6*N + 1

        # P1 = a*e -> C00 += P1
        U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00][col] += 1; col += 1
        # P2 = b*g -> C00 += P2
        U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00][col] += 1; col += 1
        # P3 = c*e -> C10 += P3
        U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10][col] += 1; col += 1
        # P4 = d*g -> C10 += P4
        U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10][col] += 1; col += 1

    # Try a lightweight rank-reduction / improvement attempt in the same family:
    # attempt to eliminate any zero columns and optionally merge columns that
    # are scalar multiples of each other (which can happen with half-integer
    # patterns in more sophisticated decompositions).  This keeps the same
    # structural family while possibly trimming the rank if any redundancy
    # was introduced.
    if time.time() < DEADLINE:
        # Identify zero columns
        rank = len(U[0])
        keep = []
        for r in range(rank):
            col_u = [U[i][r] for i in range(len(U))]
            col_v = [V[i][r] for i in range(len(V))]
            col_w = [W[i][r] for i in range(len(W))]
            if any(x != 0 for x in col_u) and any(x != 0 for x in col_v) and any(x != 0 for x in col_w):
                keep.append(r)

        if len(keep) < rank:
            U = [[U[i][r] for r in keep] for i in range(len(U))]
            V = [[V[i][r] for r in keep] for i in range(len(V))]
            W = [[W[i][r] for r in keep] for i in range(len(W))]

    return {"U": U, "V": V, "W": W}
