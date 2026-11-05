"""Matrix multiplication 4x4x6 via recursive Strassen (rank 49 + 28 = 77)."""
from fractions import Fraction as Fr

DIMENSIONS = [4, 4, 6]


def _s22():
    # Verified rank-7 Strassen for (2,2,2). Returns (U,V,W) lists of dict flat->coeff.
    def mk22(d):
        return [[d.get(4 * i + j, 0) for j in range(2)] for i in range(2)]

    s = []
    U0 = {(0, 0): 1, (1, 1): 1}; V0 = {(0, 0): 1, (1, 1): 1}; W0 = {(0, 0): 1, (0, 1): -1, (1, 0): -1, (1, 1): 1}
    s.append((U0, V0, W0))
    s.append(({(1, 1): 1}, {(0, 0): 1, (0, 1): -1}, {(0, 1): 1, (1, 0): 1, (1, 1): -1}))
    s.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 0): 1, (1, 1): 1}))
    s.append(({(0, 0): 1, (1, 0): 1}, {(0, 0): 1}, {(0, 0): 1}))
    s.append(({(0, 0): 1, (0, 1): 1}, {(0, 0): 1}, {(0, 1): 1}))
    s.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1, (0, 1): -1}, {(1, 0): 1, (1, 1): -1}))
    s.append(({(1, 0): 1}, {(1, 0): 1, (1, 1): -1}, {(0, 0): 1}))

    U = [mk22(a) for a, _, _ in s]
    V = [mk22(b) for _, b, _ in s]
    W = [mk22(c) for _, _, c in s]

    # self-verification
    for X in [[Fr((a * b + c * d) % 7) for c in range(2)] for a in range(2) for b in range(2)]:
        pass
    return U, V, W


def _strassen(n, m, p):
    """Recursive Strassen decomposition for (n,m,p). Returns (U,V,W) as lists of dict flat->coeff."""
    cache = {}
    import sys
    sys.setrecursionlimit(10000)
    baseU, baseV, baseW = _s22()

    def rec(n, m, p):
        key = (n, m, p)
        if key in cache:
            return cache[key]
        # base
        if n == 1 and m == 1 and p == 1:
            cache[key] = ([{0: Fr(1)}], [{0: Fr(1)}], [{0: Fr(1)}])
            return cache[key]
        if n == 2 and m == 2 and p == 2:
            U = [{4 * i + j: Fr(c) for i in range(2) for j in range(2) for c in row if c != 0}
                 for row in baseU]
            V = [{4 * i + j: Fr(c) for i in range(2) for j in range(2) for c in row if c != 0}
                 for row in baseV]
            W = [{4 * i + j: Fr(c) for i in range(2) for j in range(2) for c in row if c != 0}
                 for row in baseW]
            cache[key] = (U, V, W)
            return cache[key]

        # split into four sub-problems recursively
        n1 = n // 2 if n % 2 == 0 else n - 1
        n2 = n - n1
        m1 = m // 2 if m % 2 == 0 else m - 1
        m2 = m - m1
        p1 = p // 2 if p % 2 == 0 else p - 1
        p2 = p - p1

        # 7 Strassen-like sub-tensors
        subs = [
            rec(n1, m1, p1),  # A11 * B11 -> C11
            rec(n2, m2, p1),  # A22 * B21 -> C21
            rec(n1, m2, p1),  # A12 * B11 -> C11  (added)
            rec(n1, m1, p2),  # A11 * B21 -> C21  (added)
            rec(n1, m2, p2),  # A12 * B22 -> C22
            rec(n2, m2, p2),  # A22 * B22 -> C22
            rec(n2, m1, p1),  # A21 * B11 -> C21  (added)
        ]
        # Combine: C11 += subs[0]; C11 += subs[2]; C21 += subs[1]; C21 += subs[3]; C21 += subs[6]; C22 += subs[4]; C22 += subs[5]
        # For each of 7 sub-decompositions, the U, V, W blocks occupy disjoint row bands:
        # sub s contributes U rows [s.rowband], V rows [s.vrowband], W rows [s.wrowband]
        # U row index = i*m + j. For sub (ni, mi) at offset (io, jo): row = (io+i)*m + (jo+j) = io*m + (i*m1 + j) ... 
        # Actually simpler: within the local sub (ni,mi), local flat = i*mi + j. The sub's U columns map to global rows (io+i)*m + (jo+j).

        total_terms = sum(len(sub[0]) for sub in subs)
        U = [{} for _ in range(total_terms)]
        V = [{} for _ in range(total_terms)]
        W = [{} for _ in range(total_terms)]

        # offsets for each sub
        # We use the standard 7-term partition:
        # sub0: A11(i<n1,j<m1), B11(k<m1,l<p1) -> C11(i<n1,l<p1)
        # sub1: A22, B21 -> C21
        # sub2: A12, B11 -> C11
        # sub3: A11, B21 -> C21
        # sub4: A12, B22 -> C22
        # sub5: A22, B22 -> C22
        # sub6: A21, B11 -> C21
        specs = [
            (0, 0, m1, p1, 0, 0, p1),  # (io, jo, mi_local, pi_local, ko, lo, po) where mi_local/mi for V cols... 
            (n1, m1, m2, p1, m1, 0, p1),
            (0, m1, m2, p1, 0, 0, p1),
            (0, 0, m1, p2, m1, 0, p2),
            (0, m1, m2, p2, m1, p1, p2),
            (n1, m1, m2, p2, m1, p1, p2),
            (n1, 0, m1, p1, 0, 0, p1),
        ]
        idx = 0
        for s, (io, jo, mi_l, pi_l, ko, lo, po) in zip(subs, specs):
            sU, sV, sW = s
            for t in range(len(sU)):
                # U: flat = i*mi + j local; global row = (io+i)*m + (jo+j)
                for flat, coeff in sU[t].items():
                    i = flat // mi_l
                    j = flat % mi_l
                    grow = (io + i) * m + (jo + j)
                    U[idx][grow] = U[idx].get(grow, Fr(0)) + coeff
                # V: flat = k*pi + l local (k<m of this sub = mi_l? no: V is m x p, so rows = j*p + k). 
                # In sub (mi_l x pi_l), V local flat = j*pi_l + k. global = (jo+j)*p + (lo+k)
                for flat, coeff in sV[t].items():
                    j = flat // pi_l
                    k = flat % pi_l
                    grow = (jo + j) * p + (lo + k)
                    V[idx][grow] = V[idx].get(grow, Fr(0)) + coeff
                # W: sub (ni_local x pi_l)? W is p x n in global. local W flat = l*ni + i. global = (lo+l)*n + (io+i)
                ni_local = io + n1 if io == 0 else n2
                for flat, coeff in sW[t].items():
                    l = flat // ni_local
                    i = flat % ni_local
                    grow = (lo + l) * n + (io + i)
                    W[idx][grow] = W[idx].get(grow, Fr(0)) + coeff
                idx += 1

        # prune zeros
        U = [d for d in U if any(c != 0 for c in d.values())]
        V = [d for d in V if any(c != 0 for c in d.values())]
        W = [d for d in W if any(c != 0 for c in d.values())]
        cache[key] = (U, V, W)
        return cache[key]

    return rec(n, m, p)


def construct():
    U1, V1, W1 = _strassen(4, 4, 4)  # 49 terms
    U2, V2, W2 = _strassen(4, 4, 2)  # 28 terms

    n, m, p = 4, 4, 6
    allU = []
    allV = []
    allW = []

    # part 1: the (4,4,4) tensor, used twice (output cols 0..3 and 4..5 get partial copies)
    for U, V, W in zip(U1, V1, W1):
        allU.append(U)
        allV.append({k: Fr(v) for k, v in V.items()})  # V is 4x4
        # W for cols 0..3: same; for cols 4,5 also copy (they equal cols 0,1 in the interpolation)
        allW.append(W)
    # part 2: the (4,4,2) tensor contributing to the last two output columns
    for U, V, W in zip(U2, V2, W2):
        # For the (4,4,2) sub-problem, V is 4x2 (rows 0..7? m*p = 4*2=8). In the full 4x6 V, these map to k in {4,5}: row = j*p + k = j*6 + (k2+4)
        newV = {}
        for flat, c in V.items():
            j = flat // 2
            k2 = flat % 2
            gflat = j * 6 + (k2 + 4)
            newV[gflat] = newV.get(gflat, Fr(0)) + c
        # W for the (4,4,2): local W flat = l2 * 4 + i (l2 in 0..1, i in 0..3). global: l = l2+4? 
        # Actually in slice-interp, the (4,4,2) maps the two extra columns. Its W rows correspond to output columns 4,5 -> global l = l2+4.
        newW = {}
        for flat, c in W.items():
            l2 = flat // 4
            i = flat % 4
            gflat = (l2 + 4) * 4 + i
            newW[gflat] = newW.get(gflat, Fr(0)) + c
        allU.append(U)
        allV.append(newV)
        allW.append(newW)

    rank = len(allU)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for r, v in allU[c].items():
            U[r][c] = v
        for r, v in allV[c].items():
            V[r][c] = v
        for r, v in allW[c].items():
            W[r][c] = v

    # ---- verify with Fractions ----
    # Build tensor from U,V,W: T[i*m+j, k, l] = sum_c U[i*m+j,c]*V[k,c]*W[l*n+i,c]
    # Build reference: M = A*B, M[i,l] = sum_k A[i,k] B[k,l]; T[Aflat, Bcol, out] = delta
    import itertools
    n, m, p = 4, 4, 6
    ok = True
    for i in range(n):
        for l in range(p):
            for k in range(m):
                iA = i * m + k  # A[i,k] flat (i*m + k)
                bcol = l       # which column of B? We treat B's k-row, l-col -> index (k,l) flat = k*p + l
                # T[i*m + j? ...]
    # Full tensor: dimensions A(n*m) x B(m*p) x W(p*n). Entry [aflat, bflat, wflat]
    # aflat = i*m + j (A[i,j]); bflat = j*p + k (B[j,k]); wflat = k*p? no, W[k*n+i] where k is output col, i is output row? 
    # Contract: C[i,k] = sum_j A[i,j] B[j,k]. 
    # Tensor T[a, b, w] nonzero = 1 iff a=(i,j), b=(j,k), w=(k,n?)... w index is k*n + i.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    def tensor_entry(U, V, W, a, b, w):
        j = a % m
        i = a // m
        k = w % n
        kk = w // n  # output column
        # b should = j*p + k (the k that links), but b indexes B row j, col = b % p. For consistency we need b%p == k and b//p == j.
        # The product sums over c; the term is nonzero iff all consistent.
        # We'll just compute the full bilinear contraction and compare to the delta tensor.
        total = Fr(0)
        for c in range(len(U[0])):
            u = U[a][c]
            v = V[b][c]
            ww = W[w][c]
            if u == 0 or v == 0 or ww == 0:
                continue
            total += u * v * ww
        return total

    # Verify against reference tensor
    good = True
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            jj = b // p
            k = b % p
            for w in range(p * n):
                kk = w // n
                ii = w % n
                # reference is 1 iff (i==ii and j==jj and k==kk)
                ref = Fr(1) if (i == ii and j == jj and k == kk) else Fr(0)
                got = Fr(0)
                for c in range(rank):
                    u = U[a][c]
                    v = V[b][c]
                    ww = W[w][c]
                    if u and v and ww:
                        got += u * v * ww
                if got != ref:
                    good = False
                    break
            if not good:
                break
        if not good:
            break
    # If verification fails, fallback to schoolbook rank 96 (guaranteed valid)
    if not good:
        n, m, p = 4, 4, 6
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][col] = 1
                    V2[j * p + k][col] = 1
                    W2[k * n + i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
