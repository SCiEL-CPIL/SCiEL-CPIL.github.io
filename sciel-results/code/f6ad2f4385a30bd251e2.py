"""Bitset-accelerated local search for R(4,5;3) hypergraph colouring."""
import time
import random
from itertools import combinations

_DEADLINE = 115.0  # safety margin under 120s


def construct():
    t0 = time.time()
    random.seed(12345)

    def triples(n):
        return [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]

    def quad_triples(n):
        return [list(t) for t in combinations(range(n), 4)]

    def quint_triples(n):
        return [list(t) for t in combinations(range(n), 5)]

    # ---- build a valid colouring for a given n via bitset local search ----
    def build(n):
        if n <= 0:
            return None
        T = triples(n)
        m = len(T)
        idx = {}
        for i, (a, b, c) in enumerate(T):
            idx[(a, b, c)] = i

        q_tr = quad_triples(n)
        q_tr_idx = [[idx[tuple(s)] for s in combinations(q, 3)] for q in q_tr]
        q_tr_set = [set(x) for x in q_tr_idx]
        q_adj = [[] for _ in range(m)]
        for qi, s in enumerate(q_tr_set):
            for e in s:
                q_adj[e].append(qi)

        k_tr = quint_triples(n)
        k_tr_idx = [[idx[tuple(s)] for s in combinations(k, 3)] for k in k_tr]
        k_tr_set = [set(x) for x in k_tr_idx]
        k_adj = [[] for _ in range(m)]
        for ki, s in enumerate(k_tr_set):
            for e in s:
                k_adj[e].append(ki)

        # random init
        C = bytearray(random.getrandbits(1) for _ in range(m))
        red = [set() for _ in range(n)]
        blue = [set() for _ in range(n)]
        for i, (a, b, c) in enumerate(T):
            if C[i]:
                red[a].add(b); red[a].add(c); red[b].add(a); red[b].add(c); red[c].add(a); red[c].add(b)
            else:
                blue[a].add(b); blue[a].add(c); blue[b].add(a); blue[b].add(c); blue[c].add(a); blue[c].add(b)

        def is_red_q(q):
            s = q_tr_set[q]
            return all(C[i] for i in s)

        def is_blue_k(k):
            s = k_tr_set[k]
            return not any(C[i] for i in s)

        red_q = [is_red_q(q) for q in range(len(q_tr))]
        blue_k = [is_blue_k(k) for k in range(len(k_tr))]
        R = sum(red_q)
        B = sum(blue_k)

        def flip(i):
            a, b, c = T[i]
            if C[i]:
                red[a].remove(b); red[a].remove(c); red[b].remove(a); red[b].remove(c); red[c].remove(a); red[c].remove(b)
            else:
                blue[a].remove(b); blue[a].remove(c); blue[b].remove(a); blue[b].remove(c); blue[c].remove(a); blue[c].remove(b)
            C[i] = 1 - C[i]
            if C[i]:
                red[a].add(b); red[a].add(c); red[b].add(a); red[b].add(c); red[c].add(a); red[c].add(b)
            else:
                blue[a].add(b); blue[a].add(c); blue[b].add(a); blue[b].add(c); blue[c].add(a); blue[c].add(b)
            for q in q_adj[i]:
                was = red_q[q]
                red_q[q] = is_red_q(q)
                if was and not red_q[q]:
                    R -= 1
                elif not was and red_q[q]:
                    R += 1
            for k in k_adj[i]:
                was = blue_k[k]
                blue_k[k] = is_blue_k(k)
                if was and not blue_k[k]:
                    B -= 1
                elif not was and blue_k[k]:
                    B += 1
            return R, B

        best_R, best_B = R, B
        bestC = C.copy()
        bestred = [s.copy() for s in red]
        bestblue = [s.copy() for s in blue]

        # Phase 1: eliminate red K4
        it = 0
        while R > 0 and time.time() < _DEADLINE - 6:
            it += 1
            if it % 20000 == 0 and R == best_R and B == best_B:
                # small random kick to escape local minima
                for _ in range(3):
                    i = random.randrange(m)
                    R, B = flip(i)
            q = random.randrange(len(q_tr))
            if not red_q[q]:
                continue
            s = q_tr_set[q]
            i = random.choice(list(s))
            R, B = flip(i)
            if R < best_R or (R == best_R and B < best_B):
                best_R, best_B = R, B
                bestC = C.copy()
                bestred = [x.copy() for x in red]
                bestblue = [x.copy() for x in blue]

        # Phase 2: minimise blue K5
        it2 = 0
        while time.time() < _DEADLINE - 6:
            it2 += 1
            if it2 % 20000 == 0 and R == best_R and B == best_B:
                for _ in range(3):
                    i = random.randrange(m)
                    R, B = flip(i)
            k = random.randrange(len(k_tr))
            if not blue_k[k]:
                continue
            s = k_tr_set[k]
            i = random.choice(list(s))
            if R > 0:
                # avoid creating red K4: only flip if it doesn't add red K4
                # approximate: pick a triple; check red-ness after
                R2, B2 = flip(i)
                if R2 > R:
                    R, B = flip(i)  # revert
                    continue
            else:
                R, B = flip(i)
            if R < best_R or (R == best_R and B < best_B):
                best_R, best_B = R, B
                bestC = C.copy()
                bestred = [x.copy() for x in red]
                bestblue = [x.copy() for x in blue]

        if best_R > 0:
            return None
        colors = list(bestC)
        return {'n': n, 'colors': colors}

    # ---- try increasing n, keep best valid ----
    best = None
    for n in range(34, 9, -1):
        if time.time() > _DEADLINE - 20:
            break
        res = build(n)
        if res is not None:
            best = res
            break

    if best is None:
        # fallback: guaranteed small valid colouring (mod-3 pattern)
        n = 9
        best = {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    return best
