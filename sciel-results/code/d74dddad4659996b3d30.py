import numpy as np
import time

N = 16
M = N * (N - 1) // 2  # 120

# Precompute edge index for (i,j), 0<=i<j<N
_ei = np.zeros((N, N), dtype=np.int64)
_idx = 0
for i in range(N):
    for j in range(i + 1, N):
        _ei[i, j] = _idx
        _ei[j, i] = _idx
        _idx += 1

# Precompute, for each edge e=(i,j), the list of "other" edges of the three C4s
# containing e: for each common-neighbor pair (k,l) of i,j, the other 3 edges.
# A bad C4 for a fresh color on e exists iff those 3 edges have 3 distinct colors.
# Store as (i,j) -> array of 3-arrays, each a triple of edge indices.
_c4_triples = [None] * M
for i in range(N):
    for j in range(i + 1, N):
        e = int(_ei[i, j])
        neigh_i = set(range(N)) - {i, j}
        neigh_j = set(range(N)) - {i, j}
        common = [k for k in neigh_i if k in neigh_j]
        triples = []
        for a in range(len(common)):
            for b in range(a + 1, len(common)):
                k = common[a]
                l = common[b]
                # C4: i-k-j-l-i ; edges (i,k),(k,j),(j,l),(l,i)
                triples.append((int(_ei[i, k]), int(_ei[k, j]), int(_ei[j, l]), int(_ei[l, i])))
        _c4_triples[e] = triples

# For fast check: for edge e, we need for each triple (e, x, y, z) the colors of x,y,z.
# Precompute arrays of the 3 "other" edge indices per triple.
_c4_other = [None] * M
for e in range(M):
    tr = _c4_triples[e]
    if tr:
        arr = np.array([[t[1], t[2], t[3]] for t in tr], dtype=np.int64)
        _c4_other[e] = arr
    else:
        _c4_other[e] = np.zeros((0, 3), dtype=np.int64)


def _count_colors(colors):
    return len(set(colors))


def _is_valid(colors):
    """Check no C4 has 4 distinct colors."""
    for e in range(M):
        arr = _c4_other[e]
        if arr.shape[0] == 0:
            continue
        ce = colors[e]
        # colors of the 3 other edges
        cols = colors[arr]  # shape (T,3)
        # For each triple, check if the 4 (ce, a, b, c) are all distinct
        # all distinct iff ce not in {a,b,c} and a,b,c all distinct
        a, b, c = cols[:, 0], cols[:, 1], cols[:, 2]
        ab_diff = (a != b)
        bc_diff = (b != c)
        ac_diff = (a != c)
        a_eq_ce = (a == ce)
        b_eq_ce = (b == ce)
        c_eq_ce = (c == ce)
        all_distinct = (ab_diff & bc_diff & ac_diff) & (~a_eq_ce) & (~b_eq_ce) & (~c_eq_ce)
        if np.any(all_distinct):
            return False
    return True


def _greedy_build(deadline):
    colors = np.full(M, -1, dtype=np.int64)
    # Order edges: try to place fresh colors. We process edges; for each, if a fresh
    # color is feasible, assign next unused color; else reuse a safe existing color.
    used = set()
    next_color = 0
    # Edge order: use a heuristic (e.g., random-ish but deterministic) to spread colors.
    # We'll iterate edges in index order but re-check feasibility.
    # To improve, we can do multiple passes / try different orders.
    for e in range(M):
        if time.time() > deadline:
            break
        arr = _c4_other[e]
        T = arr.shape[0]
        # Try to assign a fresh color: feasible if no bad C4.
        # A fresh color c (not used anywhere yet, or even used but distinct from the 3)
        # Actually "fresh" means a new distinct color value. For it to be safe, for every
        # triple, the 3 other colors must NOT be 3 distinct (since c is new and distinct
        # from all existing). So condition: for every triple, the 3 other are not all distinct.
        if T > 0:
            cols = colors[arr]
            a, b, c = cols[:, 0], cols[:, 1], cols[:, 2]
            bad = (a != b) & (a != c) & (b != c)
            if np.any(bad):
                # Cannot use fresh color; must reuse.
                # Find an existing color that is safe: for every triple, adding it must not
                # make 4 distinct. A reused color c0 is safe iff for every triple, c0 equals
                # at least one of the 3 other colors OR the 3 others are not all distinct.
                # Simplest safe choice: pick a color that appears on one of the neighbor edges
                # of a "dangerous" triple. Easiest: reuse color of an adjacent edge.
                # We'll scan existing used colors for feasibility.
                placed = False
                # candidate colors: colors present on the 3-other edges of any triple, plus maybe 0
                cand = set()
                if T > 0:
                    cand.update(int(v) for v in cols.ravel())
                # also allow color 0 as fallback
                cand.add(0)
                for c0 in sorted(cand):
                    ok = True
                    if T > 0:
                        cols = colors[arr]
                        a, b, c = cols[:, 0], cols[:, 1], cols[:, 2]
                        others_distinct = (a != b) & (a != c) & (b != c)
                        # safe if for every triple: NOT(others_distinct AND c0 distinct from all 3)
                        c0_diff = ((a != c0) & (b != c0) & (c != c0))
                        bad = others_distinct & c0_diff
                        if np.any(bad):
                            ok = False
                            break
                    if ok:
                        colors[e] = c0
                        used.add(c0)
                        placed = True
                        break
                if not placed:
                    # ultimate fallback: color 0 (should always work as 0 may conflict but
                    # we accept; validity checked later and repaired)
                    colors[e] = 0
                    used.add(0)
            else:
                # fresh color safe
                colors[e] = next_color
                used.add(next_color)
                next_color += 1
        else:
            colors[e] = next_color
            used.add(next_color)
            next_color += 1
    # Ensure no -1 left
    for e in range(M):
        if colors[e] < 0:
            colors[e] = 0
    return colors


def _repair(colors, deadline):
    """Fix any bad C4 by recoloring one of its edges to a neighbor's color."""
    for _ in range(200):
        if time.time() > deadline:
            break
        # find a bad C4
        bad_e = -1
        found = False
        for e in range(M):
            arr = _c4_other[e]
            if arr.shape[0] == 0:
                continue
            ce = colors[e]
            cols = colors[arr]
            a, b, c = cols[:, 0], cols[:, 1], cols[:, 2]
            all_distinct = (a != b) & (a != c) & (b != c) & (a != ce) & (b != ce) & (c != ce)
            if np.any(all_distinct):
                bad_e = e
                found = True
                break
        if not found:
            break
        # recolor edge bad_e to color of one of its triple's other edges to break distinctness
        arr = _c4_other[bad_e]
        # pick first triple that is bad
        ce = colors[bad_e]
        cols = colors[arr]
        a, b, c = cols[:, 0], cols[:, 1], cols[:, 2]
        badmask = (a != b) & (a != c) & (b != c) & (a != ce) & (b != ce) & (c != ce)
        if np.any(badmask):
            t = int(np.argmax(badmask))
            # set colors[bad_e] = color of edge a[t] (so it matches one of the 3)
            colors[bad_e] = a[t]
        else:
            # shouldn't happen; break
            break
    return colors


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    # Build via greedy
    colors = _greedy_build(deadline)
    # Repair
    colors = _repair(colors, deadline)
    # Final safety: if still invalid, force a known-valid fallback (all zeros)
    if not _is_valid(colors.tolist()):
        colors = np.zeros(M, dtype=np.int64)
    # Ensure values in 0..119
    colors = np.clip(colors, 0, 119)
    return colors.tolist()
