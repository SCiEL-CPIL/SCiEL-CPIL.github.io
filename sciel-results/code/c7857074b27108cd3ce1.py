"""Matrix multiplication 3x4x7: tensor decomposition with Strassen-based blocks."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    DEADLINE = t0 + 110

    n, m, p = 3, 4, 7

    # --- Schoolbook fallback (rank 84) ---
    def _schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best = _schoolbook()

    # --- Verification ---
    def _verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # --- Strassen 2x2x2 (rank 7) ---
    _S7 = None
    def _strassen():
        global _S7
        if _S7 is not None:
            return _S7
        r = 7
        U = [[0] * r for _ in range(4)]
        V = [[0] * r for _ in range(4)]
        W = [[0] * r for _ in range(4)]
        # M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1
        # M2
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1
        # M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1
        # M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
        # M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[2][4]=1; W[0][4]=-1
        # M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        _S7 = (U, V, W)
        return _S7

    # --- Build block decomposition for a given split ---
    def _build(ns, ms, ps):
        # Compute total rank
        tot = 0
        for a in ns:
            for b in ms:
                for c in ps:
                    if a == 2 and b == 2 and c == 2:
                        tot += 7
                    else:
                        tot += a * b * c

        U = [[0] * tot for _ in range(n * m)]
        V = [[0] * tot for _ in range(m * p)]
        W = [[0] * tot for _ in range(p * n)]

        sU, sV, sW = _strassen()

        ioffs = [0]
        for x in ns[:-1]:
            ioffs.append(ioffs[-1] + x)
        joffs = [0]
        for x in ms[:-1]:
            joffs.append(joffs[-1] + x)
        koffs = [0]
        for x in ps[:-1]:
            koffs.append(koffs[-1] + x)

        col = 0
        for ii, a in enumerate(ns):
            for jj, b in enumerate(ms):
                for kk, c in enumerate(ps):
                    io, jo, ko = ioffs[ii], joffs[jj], koffs[kk]
                    if a == 2 and b == 2 and c == 2:
                        for di in range(2):
                            for dj in range(2):
                                for dc in range(7):
                                    U[(io+di)*m+(jo+dj)][col+dc] += sU[di*2+dj][dc]
                        for dj in range(2):
                            for dk in range(2):
                                for dc in range(7):
                                    V[(jo+dj)*p+(ko+dk)][col+dc] += sV[dj*2+dk][dc]
                        for dk in range(2):
                            for di in range(2):
                                for dc in range(7):
                                    W[(ko+dk)*n+(io+di)][col+dc] += sW[dk*2+di][dc]
                        col += 7
                    else:
                        for di in range(a):
                            for dj in range(b):
                                for dk in range(c):
                                    U[(io+di)*m+(jo+dj)][col] = 1
                                    V[(jo+dj)*p+(ko+dk)][col] = 1
                                    W[(ko+dk)*n+(io+di)][col] = 1
                                    col += 1
        return {"U": U, "V": V, "W": W}

    # --- Enumerate all valid splits and pick best ---
    from itertools import product

    def _all_splits(total, min_part=1):
        """Generate all compositions of 'total' into parts >= min_part."""
        if total == 0:
            yield []
            return
        for first in range(min_part, total + 1):
            for rest in _all_splits(total - first, min_part):
                yield [first] + rest

    # Precompute all candidate split triples
    n_splits = list(_all_splits(3))
    m_splits = list(_all_splits(4))
    p_splits = list(_all_splits(7))

    candidates = []
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                tot = 0
                for a in ns:
                    for b in ms:
                        for c in ps:
                            if a == 2 and b == 2 and c == 2:
                                tot += 7
                            else:
                                tot += a * b * c
                if tot < len(best["U"][0]):
                    candidates.append((tot, ns, ms, ps))

    candidates.sort()

    for tot, ns, ms, ps in candidates:
        if time.time() > DEADLINE:
            break
        try:
            res = _build(ns, ms, ps)
            if _verify(res["U"], res["V"], res["W"]):
                best = res
                # Since we sorted by rank, first valid is optimal
                break
        except Exception:
            pass

    return best
# EVOLVE-BLOCK-END
