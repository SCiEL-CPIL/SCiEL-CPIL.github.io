import time
import math
from fractions import Fraction


def _compat(assign, m, r, ignore=None):
    """Is residue r for modulus m compatible with all placed (except ignore)?"""
    for m2, r2 in assign.items():
        if m2 == ignore:
            continue
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True


def _try_place(assign, m):
    """Place m with the first compatible residue. Returns True on success."""
    for r in range(m):
        if _compat(assign, m, r):
            assign[m] = r
            return True
    return False


def _valid(assign):
    items = list(assign.items())
    for i in range(len(items)):
        m1, r1 = items[i]
        for j in range(i + 1, len(items)):
            m2, r2 = items[j]
            if r1 % math.gcd(m1, m2) == r2:
                return False
    return True


def construct():
    t0 = time.time()
    DEADLINE = 100.0
    MIN_M, MAX_M = 20, 200
    moduli = list(range(MIN_M, MAX_M + 1))
    assign = {}

    # ---- Phase 1: greedy forward (increasing m = highest weight first) ----
    for m in moduli:
        _try_place(assign, m)

    def unplaced():
        return [m for m in moduli if m not in assign]

    # ---- Phase 2: local search to add skipped (smaller) moduli ----
    #   by reassigning larger (lower-weight) placed moduli.
    ALT_CAP = 60
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for m in sorted(unplaced()):
            if _try_place(assign, m):
                improved = True
                continue
            # one-step reassignment: move a larger placed modulus p to free a slot
            for p, r2 in list(assign.items()):
                if p <= m:
                    continue
                gp = math.gcd(m, p)
                # only worth moving p if it actually blocks m in some way
                alt_tries = 0
                for alt in range(p):
                    if alt == r2:
                        continue
                    alt_tries += 1
                    if alt_tries > ALT_CAP:
                        break
                    if not _compat(assign, p, alt, ignore=m):
                        continue
                    assign[p] = alt
                    if _try_place(assign, m):
                        improved = True
                        break
                    assign[p] = r2  # revert
                if m in assign:
                    break

    # ---- Phase 3: swap pass (drop a larger placed p to add a smaller skipped m) ----
    if time.time() < DEADLINE:
        for m in sorted(unplaced()):
            for p in [q for q in assign if q > m]:
                saved = assign.pop(p)
                if _try_place(assign, m):
                    # keep the swap (m has higher weight than p)
                    pass
                else:
                    assign[p] = saved
                    break

    # ---- Phase 4: final verification (drop any accidental conflict) ----
    items = sorted(assign.items())
    final = {}
    for m, r in items:
        if _compat(final, m, r):
            final[m] = r

    result = [[m, r] for m, r in sorted(final.items())]
    return result
