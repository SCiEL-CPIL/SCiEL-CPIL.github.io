"""Worst interval starts for distinct-multiple matching.

We want an integer m in [0, 1000000] maximizing f(12, m): the least positive
L such that the open interval (m, m+L] contains 12 distinct integers b_i with
i | b_i.  f depends only on m mod P where P = lcm(1..12) = 27720, so we can
brute-force all residues with an exact bipartite-matching test.
"""
import time

_DEADLINE = time.time() + 80.0          # safety margin under the 90 s kill
_P = 27720                              # lcm(1..12)


def _lcm(a, b):
    return a // _gcd(a, b) * b


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _has_perfect_matching(rem, n):
    """Exact bipartite matching: can we assign distinct targets to 1..n?

    target t is usable for index i iff (rem + t) % i == 0.
    A perfect matching exists iff the greedy Hall test over the target set
    succeeds (a matching of size n into n targets).
    """
    # For each target t (1..n) compute the set of indices that can use it.
    # We run a standard augmenting-path (Kuhn) matching on the index side.
    match_t = [-1] * (n + 1)  # which index is matched to target t
    for i in range(1, n + 1):
        seen = [False] * (n + 1)
        if not _dfs(i, seen, match_t, rem, n):
            return False
    return True


def _dfs(i, seen, match_t, rem, n):
    for t in range(1, n + 1):
        if seen[t]:
            continue
        if (rem + t) % i != 0:
            continue
        seen[t] = True
        if match_t[t] == -1 or _dfs(match_t[t], seen, match_t, rem, n):
            match_t[t] = i
            return True
    return False


def construct():
    # f(12, m) is at most 144 by the contract, so the least positive L is in 1..144.
    # For a given residue r (0..P-1), f = min L in 1..144 with a perfect matching.
    # We want to MAXIMIZE f, so we want residues where the matching only appears
    # at large L.  Evaluate all residues; keep the one with the largest f, and
    # among ties pick the smallest m in [0, 1000000] (any valid m is fine, but a
    # concrete small one is safe).  Symmetry: f(r) == f(P - r) is NOT generally
    # true, so we just scan all P residues (P = 27720 is small).

    best_r = 0
    best_f = -1
    for r in range(_P):
        # find least L in 1..144 with a perfect matching
        f = 145  # default: none found (shouldn't happen; L=144 always works)
        for L in range(1, 145):
            if time.time() > _DEADLINE:
                # return best-so-far mapped into [0, 1000000]
                return int(best_r % 1000001)
            if _has_perfect_matching(r, 12):
                f = L
                break
        if f > best_f:
            best_f = f
            best_r = r
    # Map best residue into a concrete m in [0, 1000000]
    m = int(best_r)  # best_r is in 0..P-1 which is within range
    if not isinstance(m, int):
        m = int(m)
    return m
