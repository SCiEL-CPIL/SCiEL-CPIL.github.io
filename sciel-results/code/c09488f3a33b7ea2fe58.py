# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z containing no two elements whose difference
    is a nonzero square modulo 205.

    Structural facts:
      - 205 = 5 * 41. By CRT, a difference d is a nonzero square mod 205 iff
        d is a nonzero square mod 5 AND d is a nonzero square mod 41.
      - Squares mod 5: {1, 4} (2 values).
      - Squares mod 41: 20 nonzero values (41 = 1 mod 4).
      - Hence the forbidden difference set F has exactly 20 elements, and the
        conflict graph (vertices 0..204, edges = forbidden differences) is 20-regular.

    Method:
      1. Build the 20-regular conflict graph with per-vertex bitmasks (205 bits)
         for O(1) conflict tests.
      2. Greedy construction (many random orderings) for initial independent sets.
      3. Iterated local search: single-add, swap, and random kick moves,
         with a wall-clock deadline guard.
      4. Return the best feasible set found.

    Returns:
        np.ndarray of shape (s,), dtype int64, distinct residues in 0..204 with
        no forbidden pair.
    """
    m = 205
    deadline = time.time() + 570.0  # safety margin under the 630s kill

    # Forbidden differences: nonzero squares mod 205.
    forbidden = set()
    for x in range(1, m):
        v = (x * x) % m
        if v != 0:
            forbidden.add(v)
    F = sorted(forbidden)

    # Build adjacency bitmask per vertex.
    adj = [0] * m
    for v in range(m):
        mask = 0
        for d in F:
            u1 = (v + d) % m
            u2 = (v - d) % m
            mask |= (1 << u1)
            mask |= (1 << u2)
        adj[v] = mask

    def can_add(c, S):
        return (adj[c] & S) == 0

    def greedy(order, base=0):
        S = base
        for c in order:
            if (S >> c) & 1:
                continue
            if can_add(c, S):
                S |= (1 << c)
        return S

    def bits_to_list(S):
        out = []
        i = 0
        while S:
            if S & 1:
                out.append(i)
            S >>= 1
            i += 1
        return out

    def popcount(S):
        return bin(S).count('1')

    def neighbors_of(S):
        U = 0
        i = 0
        tmp = S
        while tmp:
            if tmp & 1:
                U |= adj[i]
            tmp >>= 1
            i += 1
        return U

    def try_add(S, sz):
        U = neighbors_of(S)
        for v in range(m):
            if ((S >> v) & 1) or ((U >> v) & 1):
                continue
            return S | (1 << v), sz + 1, True
        return S, sz, False

    def try_swap(S, sz):
        verts = bits_to_list(S)
        for a in verts:
            S_minus = S & ~(1 << a)
            U = neighbors_of(S_minus)
            added = 0
            for v in range(m):
                if ((S_minus >> v) & 1) or ((U >> v) & 1):
                    continue
                S_minus |= (1 << v)
                U |= adj[v]
                added += 1
                if added >= 2:
                    break
            if added >= 1:
                sz2 = popcount(S_minus)
                if sz2 > sz:
                    return S_minus, sz2, True
        return S, sz, False

    rng = random.Random(205)
    best_S = 0
    best_size = 0

    # Phase 1: many random greedy starts.
    for _ in range(20000):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        S = greedy(order)
        sz = popcount(S)
        if sz > best_size:
            best_size = sz
            best_S = S
        if best_size >= 17:
            break

    # Phase 2: iterated local search.
    S = best_S
    sz = best_size
    stall = 0
    while time.time() < deadline:
        improved = False

        S, sz, ok = try_add(S, sz)
        if ok:
            improved = True
            if sz > best_size:
                best_size = sz
                best_S = S

        S, sz, ok = try_swap(S, sz)
        if ok:
            improved = True
            if sz > best_size:
                best_size = sz
                best_S = S

        if not improved:
            stall += 1
            verts = bits_to_list(S)
            if len(verts) > 2:
                k = min(3, len(verts) - 1)
                to_remove = rng.sample(verts, k)
                S2 = S
                for a in to_remove:
                    S2 &= ~(1 << a)
                order = list(range(m))
                rng.shuffle(order)
                S2 = greedy(order, base=S2)
                sz2 = popcount(S2)
                if sz2 > sz:
                    S, sz = S2, sz2
                    if sz > best_size:
                        best_size = sz
                        best_S = S
            if stall > 200:
                break

    # Final safety: verify best_S is feasible (defensive).
    bl = bits_to_list(best_S)
    Fset = set(F)
    for i in range(len(bl)):
        for j in range(i + 1, len(bl)):
            d = (bl[i] - bl[j]) % m
            if d in Fset:
                # should not happen; fall back to a known-good greedy
                order = list(range(m))
                rng.shuffle(order)
                bl = bits_to_list(greedy(order))
                break
        else:
            continue
        break

    return np.array(bl, dtype=np.int64)


# EVOLVE-BLOCK-END
