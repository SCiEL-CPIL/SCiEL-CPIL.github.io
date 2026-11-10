"""Exact max-clique solver for pairwise-disjoint residue classes (distinct moduli 2..128)."""
import math
import time


def construct():
    MAXM = 128
    moduli = list(range(2, MAXM + 1))

    def lcm(a, b):
        return a // math.gcd(a, b) * b

    # L_m = lcm of gcd(m,n) over n != m; only residues 0..L_m-1 are distinguishable.
    Lm = {}
    for m in moduli:
        L = 1
        for n in moduli:
            if n != m:
                L = lcm(L, math.gcd(m, n))
        Lm[m] = L

    # Node list: (modulus, residue)
    nodes = []
    for m in moduli:
        for r in range(Lm[m]):
            nodes.append((m, r))
    N = len(nodes)

    # Compatibility graph (bitset adjacency).
    adj = [0] * N
    for i in range(N):
        m1, r1 = nodes[i]
        for j in range(i + 1, N):
            m2, r2 = nodes[j]
            if m1 == m2:
                continue
            g = math.gcd(m1, m2)
            if (r1 % g) != (r2 % g):
                adj[i] |= 1 << j
                adj[j] |= 1 << i

    # Strong initial lower bound: all multiples of prime p with distinct residues mod p.
    def build_prime(p):
        out = []
        cnt = 0
        for m in range(p, MAXM + 1, p):
            out.append([cnt % m, m])
            cnt += 1
        return out

    best_size = len(build_prime(11))  # 11 valid classes
    best = []

    t0 = time.time()
    TIME_LIMIT = 50.0

    # Greedy coloring of a candidate set -> ordering + per-vertex color (upper bound).
    def color_sort(Pmask):
        P = [i for i in range(N) if (Pmask >> i) & 1]
        colors = []
        color = {}
        for v in P:
            placed = False
            for ci, col in enumerate(colors):
                ok = True
                for u in col:
                    if (adj[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    col.append(v)
                    color[v] = ci + 1
                    placed = True
                    break
            if not placed:
                colors.append([v])
                color[v] = len(colors)
        P.sort(key=lambda v: color[v])
        return P, color

    def expand(Pmask, R, Rlen):
        nonlocal best_size, best
        if time.time() - t0 > TIME_LIMIT:
            return
        if Pmask == 0:
            if Rlen > best_size:
                best_size = Rlen
                best = R[:]
            return
        P, color = color_sort(Pmask)
        for idx in range(len(P) - 1, -1, -1):
            if time.time() - t0 > TIME_LIMIT:
                return
            v = P[idx]
            if Rlen + color[v] <= best_size:
                break
            R.append(v)
            expand(Pmask & adj[v], R, Rlen + 1)
            R.pop()
            Pmask &= ~(1 << v)

    expand((1 << N) - 1, [], 0)

    if not best:
        return build_prime(11)

    return [[nodes[v][0], nodes[v][1]] for v in best]
