"""Structural rewrite: branch-and-bound maximum independent set on the
conflict graph (Tomita-style), with bitsets and a greedy coloring bound.

A pair {a,b} is a "conflict" if there exists c in [1,128] such that
lcm(a,b)=lcm(a,c)=lcm(b,c).  A valid set is an independent set in this graph.
We maximize |independent set| via exact branch-and-bound (fast for n=128).
"""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 80.0

    def _lcm(a, b):
        return a // gcd(a, b) * b

    n = 128

    # ---- Build conflict graph (pair is conflict iff a common bad third exists)
    adj = [0] * (n + 1)          # bit i (0-based) set if (i+1) conflicts with x
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    adj[a] |= 1 << (b - 1)
                    adj[b] |= 1 << (a - 1)
                    break

    # ---- Greedy coloring bound on a candidate set (bitmask)
    def color_bound(cands):
        order = []
        while cands:
            lb = cands & (-cands)
            order.append(lb)
            cands ^= lb
        color_of = {}
        next_color = 1
        for bit in order:
            used = 0
            for b2 in order:
                if b2 == bit:
                    break
                if (adj[bit.bit_length()] >> (b2.bit_length() - 1)) & 1:
                    used |= 1 << color_of[b2]
            c = 1
            while used & (1 << (c - 1)):
                c += 1
            color_of[bit] = c
            if c > next_color:
                next_color = c
        return next_color

    # ---- Initial greedy independent set (lower bound)
    def greedy_seed(order):
        chosen = set()
        for x in order:
            ok = True
            for y in chosen:
                if (adj[x] >> (y - 1)) & 1:
                    ok = False
                    break
            if ok:
                chosen.add(x)
        return chosen

    best = greedy_seed(range(1, n + 1))
    for order in (range(n, 0, -1),
                  sorted(range(1, n + 1), key=lambda x: bin(adj[x]).count('1'))):
        s = greedy_seed(order)
        if len(s) > len(best):
            best = s
    best_size = len(best)

    # ---- Branch and bound (Tomita-style)
    def expand(cands, size):
        nonlocal best, best_size
        if time.time() > deadline - 2.0:
            return
        if size + color_bound(cands) <= best_size:
            return
        while cands:
            lb = cands & (-cands)
            x = lb.bit_length()
            cands ^= lb
            if size + 1 + color_bound(cands) > best_size:
                expand(cands & ~adj[x], size + 1)
            if size + color_bound(cands) <= best_size:
                return

    if time.time() < deadline - 3.0:
        expand((1 << n) - 1, 0)

    # ---- Local search: 1-for-2 swaps to push past the B&B incumbent
    def conflicts_with(x, S):
        for y in S:
            if (adj[x] >> (y - 1)) & 1:
                return True
        return False

    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for x in range(1, n + 1):
            if x not in best and not conflicts_with(x, best):
                best.add(x)
                best_size += 1
                improved = True
                break

    for _ in range(200):
        if time.time() > deadline - 3.0:
            break
        found = False
        for x in list(best):
            trial = best - {x}
            gain = 0
            for y in range(1, n + 1):
                if y == x or y in trial:
                    continue
                if not conflicts_with(y, trial):
                    trial.add(y)
                    gain += 1
                    if gain >= 2:
                        break
            if len(trial) > len(best):
                best = trial
                best_size = len(trial)
                found = True
                break
        if not found:
            break

    return list(best)
# EVOLVE-BLOCK-END
