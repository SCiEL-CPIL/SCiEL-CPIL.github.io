import random
import time

_N = 16
_EDGES = 120
_DEADLINE = 138.0  # safety margin before the 150s kill


def _build_edge_index():
    idx = {}
    k = 0
    for i in range(_N):
        for j in range(i + 1, _N):
            idx[(i, j)] = k
            k += 1
    return idx


_EDGE_INDEX = _build_edge_index()


def _build_k4():
    k4 = []
    for a in range(_N):
        for b in range(a + 1, _N):
            for c in range(b + 1, _N):
                for d in range(c + 1, _N):
                    k4.append((a, b, c, d))
    return k4


_K4 = _build_k4()
_K4_E = [[_EDGE_INDEX[(a, b)], _EDGE_INDEX[(a, c)], _EDGE_INDEX[(a, d)],
          _EDGE_INDEX[(b, c)], _EDGE_INDEX[(b, d)], _EDGE_INDEX[(c, d)]]
         for (a, b, c, d) in _K4]


def _initial_colors():
    # (i+j) mod 17: valid, 17 colours.
    cols = [0] * _EDGES
    for i in range(_N):
        for j in range(i + 1, _N):
            cols[_EDGE_INDEX[(i, j)]] = (i + j) % 17
    return cols


def _worst(cols):
    """Return (worst_count, worst_k4_index)."""
    worst = 6
    wi = -1
    for t, e in enumerate(_K4_E):
        c0, c1, c2, c3, c4, c5 = (cols[e[0]], cols[e[1]], cols[e[2]],
                                   cols[e[3]], cols[e[4]], cols[e[5]])
        s = {c0, c1, c2, c3, c4, c5}
        if len(s) < worst:
            worst = len(s)
            wi = t
    return worst, wi


def _improve(cols, rng):
    """Local search: fix colour collisions in worst K4s, minimise palette."""
    deadline = time.time() + _DEADLINE
    best = _initial_colors()
    bw, _ = _worst(best)
    if bw >= 5:
        return best
    cur = best[:]
    cur_w, cur_i = bw, 0
    stall = 0
    while time.time() < deadline - 0.5:
        # ensure a worst K4 is known
        if cur_i < 0:
            cur_w, cur_i = _worst(cur)
        if cur_w >= 5:
            break
        e = _K4_E[cur_i]
        c0, c1, c2, c3, c4, c5 = (cur[e[0]], cur[e[1]], cur[e[2]],
                                   cur[e[3]], cur[e[4]], cur[e[5]])
        s = {c0, c1, c2, c3, c4, c5}
        if len(s) >= 5:
            cur_i = -1
            continue
        # find the most duplicated colour in this K4
        cnt = {}
        for v in (c0, c1, c2, c3, c4, c5):
            cnt[v] = cnt.get(v, 0) + 1
        dup = max(cnt, key=lambda x: cnt[x])
        # candidate edges: those carrying dup
        cand = [e[k] for k in range(6) if (c0, c1, c2, c3, c4, c5)[k] == dup]
        if not cand:
            cand = list(e)
        rng.shuffle(cand)
        improved = False
        for ei in cand:
            newc = rng.randrange(0, 60)
            if newc == cur[ei]:
                continue
            old = cur[ei]
            cur[ei] = newc
            # recompute worst
            nw, ni = _worst(cur)
            if nw > cur_w or (nw == cur_w and nw >= 5):
                cur_w, cur_i = nw, ni
                improved = True
                break
            cur[ei] = old
        if improved:
            stall = 0
            if cur_w >= 5:
                break
        else:
            stall += 1
            cur_i = -1
            if stall > 400:
                # random kick
                for _ in range(20):
                    cur[rng.randrange(_EDGES)] = rng.randrange(0, 60)
                cur_w, cur_i = _worst(cur)
                stall = 0
    # palette minimisation on a valid (or best) coloring
    if cur_w >= 5:
        best = cur
    else:
        best = cur  # keep best effort
    # minimise distinct colours
    used = sorted({v for v in best if v >= 0})
    remap = {}
    for new, old in enumerate(used):
        remap[old] = new
    best = [remap.get(v, 0) for v in best]
    return best


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    rng = random.Random(20240517)
    best = _initial_colors()
    bw, _ = _worst(best)
    # keep the best valid (>=5) coloring seen
    best_valid = best if bw >= 5 else None
    best_valid_w = bw

    attempts = 0
    while time.time() < deadline - 1.0:
        attempts += 1
        res = _improve(best, rng)
        w, _ = _worst(res)
        if w > best_valid_w:
            best_valid = res
            best_valid_w = w
            if best_valid_w >= 5:
                break
        # feed back
        best = res
        if best_valid_w >= 5:
            break

    final = best_valid if best_valid is not None else best
    # remap to compact palette
    used = sorted({v for v in final})
    remap = {old: new for new, old in enumerate(used)}
    final = [remap.get(v, 0) for v in final]
    return final
