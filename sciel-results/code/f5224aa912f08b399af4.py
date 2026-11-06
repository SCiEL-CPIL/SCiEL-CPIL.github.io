"""XOR-circuit synthesis: shared low-weight subexpression pool + greedy/exact row cover."""
import time
import sys
from collections import defaultdict

# ---- global deadline guard (wall-clock) ----
_T0 = time.time()
_DEADLINE = 90.0          # evaluator kills at 90s
_MARGIN = 8.0             # safety margin -> effective budget
_BUDGET = _DEADLINE - _MARGIN

def _time_left():
    return _BUDGET - (time.time() - _T0)


def _row_supports(matrix):
    """Return list of frozensets (one per row) of input indices with a 1."""
    supports = []
    for r in range(len(matrix)):
        row = matrix[r]
        s = frozenset(j for j in range(len(row)) if row[j])
        supports.append(s)
    return supports


def _subexpr_weight(s):
    return len(s)


def _pool_greedy_setcover(supports, max_pool_weight, time_budget):
    """
    Build a shared pool of low-weight subexpressions (frozensets) that are useful
    for covering the column demands. Greedy: repeatedly add the candidate
    subexpression (weight <= max_pool_weight) that appears in the most remaining
    uncovered row-support 'needs'.

    We model the need as: each row-support S must be expressible as XOR of pool
    elements. A proxy for 'useful' is the count of rows whose support contains
    this subexpression (so it can reduce the row). We keep it simple and effective:
    seed pool with all weight-1 and weight-2 sets that actually appear as subsets
    of some row support, then greedily add higher-weight sets that are subsets of
    remaining (not-yet-fully-covered) row supports.
    """
    pool = set()

    # candidate generation from row supports
    from itertools import combinations
    cand = set()
    for S in supports:
        if not S:
            continue
        # weight 1
        for a in S:
            cand.add(frozenset([a]))
        # weight 2
        for a, b in combinations(sorted(S), 2):
            cand.add(frozenset([a, b]))
        # weight 3 (only if not too many)
        if len(S) <= max_pool_weight + 2:
            for a, b, c in combinations(sorted(S), 3):
                if 3 <= max_pool_weight:
                    cand.add(frozenset([a, b, c]))
        # weight 4
        if len(S) <= max_pool_weight + 2:
            for a, b, c, d in combinations(sorted(S), 4):
                if 4 <= max_pool_weight:
                    cand.add(frozenset([a, b, c, d]))

    # seed: all weight-1 and weight-2 candidates (cheap, always useful)
    for c in cand:
        w = len(c)
        if w <= 2:
            pool.add(c)

    # greedy add higher-weight sets that are subsets of some row support
    remaining_cands = [c for c in cand if len(c) > 2]

    # count of rows that 'need' a candidate = number of row supports that contain it
    # and are not yet fully representable (approx: still have residual weight > 0)
    # We approximate residual by whether the row support is in pool or xor-combinable.
    # For speed, use containment frequency as the greedy score.

    # Precompute containment counts efficiently via row-support membership
    # Build for each candidate the number of row supports that contain it.
    # Use a map from frozenset -> count of rows containing it.
    contain = defaultdict(int)
    for S in supports:
        elems = sorted(S)
        n = len(elems)
        # enumerate subsets up to max_pool_weight
        for k in range(1, min(n, max_pool_weight) + 1):
            for sub in combinations(elems, k):
                fs = frozenset(sub)
                contain[fs] += 1

    # iterate greedy
    for _ in range(200):  # bounded
        if _time_left() < 2.0:
            break
        # pick candidate (not in pool) with max contain count
        best = None
        best_cnt = 0
        for c in remaining_cands:
            if c in pool:
                continue
            cnt = contain.get(c, 0)
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
        if best is None or best_cnt <= 1:
            break
        pool.add(best)
        remaining_cands.remove(best)

    return pool


def _xor_combine(supports, pool, time_budget):
    """
    Express each row support as XOR of pool elements.
    Returns:
      gates: list of (wire_a, wire_b) where gate k produces wire n+1+k
      outputs: list of wire indices per row
      wire_of: dict frozenset -> wire index
    Wire convention: input i -> wire i (0..n-1), zero -> n, gate k -> n+1+k.
    """
    n = max(len(s) for s in supports) if supports else 0
    # wire_of: map frozenset -> wire index
    wire_of = {}
    gates = []

    def get_wire(fs):
        """Ensure subexpression fs exists as a wire; return its wire index."""
        if fs in wire_of:
            return wire_of[fs]
        # if it's a single input
        if len(fs) == 1:
            (a,) = next(iter(fs))
            w = a
            wire_of[fs] = w
            return w
        if len(fs) == 0:
            w = n  # zero wire
            wire_of[fs] = w
            return w
        # need to build it from pool elements via XOR
        # find a decomposition of fs as XOR of pool elements (greedy + exact small)
        decomp = _decompose(fs, pool, wire_of, gates, n)
        if decomp is None:
            # fallback: build directly from inputs (may be heavy but correct)
            elems = sorted(fs)
            w = elems[0]
            for e in elems[1:]:
                # XOR with input e
                a = w
                b = e
                gates.append((a, b))
                w = n + 1 + (len(gates) - 1)
            wire_of[fs] = w
            return w
        wire_of[fs] = w
        return w

    def _decompose(fs, pool, wire_of, gates, n):
        # Try to express fs as XOR of <= 2 pool elements first (fast path)
        # 1 element
        if fs in pool:
            return [fs]
        # 2 elements: find p in pool such that fs ^ p in pool
        # (XOR of supports = symmetric difference)
        for p in pool:
            q = fs ^ p
            if q in pool:
                return [p, q]
        # 3 elements via pair
        # precompute pool as list
        plist = list(pool)
        for i in range(len(plist)):
            p = plist[i]
            rem = fs ^ p
            if not rem:
                return [p]
            for j in range(i + 1, len(plist)):
                q = plist[j]
                r = rem ^ q
                if r in pool:
                    return [p, q, r]
        # fallback greedy: repeatedly pick pool element that is subset of rem
        rem = fs
        chosen = []
        while rem:
            best = None
            for p in pool:
                if p and p <= rem and (len(p) > len(best) if best else True):
                    best = p
            if best is None:
                # no pool subset; pick any pool element overlapping rem to reduce
                for p in pool:
                    if p & rem:
                        best = p
                        break
            if best is None:
                return None
            chosen.append(best)
            rem = rem ^ best
            if len(chosen) > 16:
                return None
        return chosen

    outputs = []
    for S in supports:
        if not S:
            outputs.append(n)  # zero
            continue
        # try to use pool decomposition to minimize gates, reusing existing wires
        # get wire for the full support (will create gates as needed, sharing)
        w = get_wire(S)
        outputs.append(w)

    return gates, outputs


def _verify(gates, outputs, matrix, n):
    """Exact GF(2) verification. Return True if outputs match matrix rows."""
    # compute support (frozenset) for each wire
    sup = {}
    for i in range(n):
        sup[i] = frozenset([i])
    sup[n] = frozenset()  # zero
    for k, (a, b) in enumerate(gates):
        w = n + 1 + k
        sup[w] = sup[a] ^ sup[b]
    for r, w in enumerate(outputs):
        expected = frozenset(j for j in range(len(matrix[r])) if matrix[r][j])
        if sup.get(w) != expected:
            return False
    return True


def synthesize(matrix, depth_limit):
    """
    Interface: synthesize(matrix, depth_limit) -> {'gates': [[wire, wire], ...],
    'outputs': [wire, ...]}
    Input wires 0..n-1, constant-zero wire n, gate k -> wire n+1+k.
    """
    # normalize matrix to list of lists of ints
    m = len(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}
    n = len(matrix[0])
    supports = _row_supports(matrix)

    # If all rows are zero
    if all(not s for s in supports):
        return {'gates': [], 'outputs': [n] * m}

    # Deadline guard
    if _time_left() < 1.0:
        # emergency: build each row directly from inputs
        gates = []
        outputs = []
        for s in supports:
            if not s:
                outputs.append(n)
                continue
            elems = sorted(s)
            w = elems[0]
            for e in elems[1:]:
                gates.append((w, e))
                w = n + 1 + (len(gates) - 1)
            outputs.append(w)
        return {'gates': [list(g) for g in gates], 'outputs': outputs}

    # Build shared pool via greedy set cover
    max_pool_weight = 4
    pool = _pool_greedy_setcover(supports, max_pool_weight, _time_left())

    # Combine rows using pool with sharing
    gates, outputs = _xor_combine(supports, pool, _time_left())

    # Verify; if fails, fall back to direct construction
    if not _verify(gates, outputs, matrix, n):
        gates = []
        outputs = []
        for s in supports:
            if not s:
                outputs.append(n)
                continue
            elems = sorted(s)
            w = elems[0]
            for e in elems[1:]:
                gates.append((w, e))
                w = n + 1 + (len(gates) - 1)
            outputs.append(w)

    return {'gates': [list(g) for g in gates], 'outputs': outputs}
