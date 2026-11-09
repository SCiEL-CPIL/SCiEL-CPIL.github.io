import random
import time
from itertools import combinations

N = 16
E = 120

def _edge_idx(i, j):
    return i * 15 - i * (i - 1) // 2 + (j - i - 1)

# Precompute all K4s (6 edge indices each)
_K4s = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                _K4s.append((
                    _edge_idx(a, b), _edge_idx(a, c), _edge_idx(a, d),
                    _edge_idx(b, c), _edge_idx(b, d), _edge_idx(c, d),
                ))
_NUM_K4 = len(_K4s)

# For each edge, list of (k4_index, slot)
_EDGE_K4 = [[] for _ in range(E)]
for _k, _k4 in enumerate(_K4s):
    for _s, _e in enumerate(_k4):
        _EDGE_K4[_e].append((_k, _s))

def _init_colors():
    # 16-coloring: colour (i,j) = (i+j) mod 16  -> uses colours 0..15 (16 distinct)
    cols = [0] * E
    for i in range(N):
        for j in range(i + 1, N):
            cols[_edge_idx(i, j)] = (i + j) % 16
    return cols

def _validate(cols):
    for k4 in _K4s:
        if len({cols[e] for e in k4}) < 5:
            return False
    return True

def _full_score(cols):
    bad = 0
    for k4 in _K4s:
        if len({cols[e] for e in k4}) < 5:
            bad += 1
    return bad, len(set(cols))

def _delta(e, oldc, newc, cols, masks, bad):
    """Return (new_bad, new_used) after hypothetically setting cols[e]=newc."""
    if oldc == newc:
        return bad, len(set(cols))
    delta_bad = 0
    used = set(cols)
    if used and oldc in used and sum(1 for x in cols if x == oldc) == 1:
        used.discard(oldc)
    used.add(newc)
    for (k, s) in _EDGE_K4[e]:
        m = masks[k]
        ob = (m >> (s * 8)) & 0xFF
        if ob == oldc:
            m &= ~(0xFF << (s * 8))
            m |= (newc << (s * 8))
            if len({(m >> (t * 8)) & 0xFF for t in range(6)}) < 5:
                if ob != newc:
                    delta_bad += 1
            else:
                if ob != newc:
                    delta_bad -= 1
            masks[k] = m
    return bad + delta_bad, len(used)

def construct():
    start = time.time()
    deadline = start + 145.0
    rng = random.Random(12345)

    cols = _init_colors()
    bad, used = _full_score(cols)
    masks = [0] * _NUM_K4
    for k, k4 in enumerate(_K4s):
        m = 0
        for s, e in enumerate(k4):
            m |= (cols[e] << (s * 8))
        masks[k] = m

    best = cols[:]
    best_score = (bad, used)

    def consider(nc, nb, nu):
        nonlocal best, best_score
        if (nb, nu) < best_score:
            best = nc[:]
            best_score = (nb, nu)

    def time_left():
        return deadline - time.time()

    def local_search(max_steps):
        nonlocal cols, bad, used, masks, best, best_score
        for _ in range(max_steps):
            if time_left() < 0.5:
                return
            if bad == 0:
                return
            e = rng.randrange(E)
            oldc = cols[e]
            # candidate colours: existing colours + a few random low colours
            cand = list(set(cols))
            for c in range(0, 12):
                if c not in cand:
                    cand.append(c)
            rng.shuffle(cand)
            improved = False
            for newc in cand:
                if newc == oldc:
                    continue
                nb, nu = _delta(e, oldc, newc, cols, masks, bad)
                if nb < bad:
                    cols[e] = newc
                    bad, used = nb, nu
                    consider(cols, bad, used)
                    improved = True
                    break
                elif nb == bad and nu < used:
                    cols[e] = newc
                    bad, used = nb, nu
                    consider(cols, bad, used)
                    improved = True
                    break
            if not improved and bad == 0:
                return

    # Phase 1: eliminate bad K4s (keep ~16 colours)
    local_search(200000)

    # Phase 2: reduce number of colours
    for target in (15, 14, 13):
        if time_left() < 2.0:
            break
        if used <= target and bad == 0:
            continue
        # cap colours: remap anything >= target down
        for e in range(E):
            if cols[e] >= target:
                cols[e] = rng.randrange(target)
        bad, used = _full_score(cols)
        for k, k4 in enumerate(_K4s):
            m = 0
            for s, e2 in enumerate(k4):
                m |= (cols[e2] << (s * 8))
            masks[k] = m
        consider(cols, bad, used)
        # local search restricted to colours < target
        for _ in range(300000):
            if time_left() < 1.0:
                break
            if bad == 0 and used <= target:
                break
            e = rng.randrange(E)
            oldc = cols[e]
            improved = False
            for _t in range(8):
                newc = rng.randrange(target)
                if newc == oldc:
                    continue
                nb, nu = _delta(e, oldc, newc, cols, masks, bad)
                if nb < bad or (nb == bad and nu < used):
                    cols[e] = newc
                    bad, used = nb, nu
                    consider(cols, bad, used)
                    improved = True
                    break
            if not improved and bad == 0 and used <= target:
                break

    # Final: return best valid if any, else current best
    if best_score[0] == 0:
        return best
    if bad == 0:
        return cols
    # fallback: return the seed (always valid)
    return _init_colors()
