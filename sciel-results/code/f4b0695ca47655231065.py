"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _overlap(a, b):
    # a, b are (x, y, z, w, h, d)
    ax, ay, az, aw, ah, ad = a
    bx, by, bz, bw, bh, bd = b
    if ax + aw <= bx or bx + bw <= ax:
        return False
    if ay + ah <= by or by + bh <= ay:
        return False
    if az + ad <= bz or bz + bd <= az:
        return False
    return True


def _fits_in_bin(x, y, z, w, h, d, W, H, D, placed):
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlap((x, y, z, w, h, d), (px, py, pz, pw, ph, pd)):
            return False
    return True


def _candidate_positions(W, H, D, placed, w, h, d):
    # Generate candidate corner positions, ordered by (z, y, x) preference.
    xs = sorted(set([0] + [p[0] + p[3] for p in placed] + [p[0] for p in placed]))
    ys = sorted(set([0] + [p[1] + p[4] for p in placed] + [p[1] for p in placed]))
    zs = sorted(set([0] + [p[2] + p[5] for p in placed] + [p[2] for p in placed]))
    cands = []
    for z in zs:
        if z + d > D:
            continue
        for y in ys:
            if y + h > H:
                continue
            for x in xs:
                if x + w > W:
                    continue
                cands.append((z, y, x))
    # also include pure origin and "max-extent" corners for completeness
    cands.append((0, 0, 0))
    cands.append((max((p[2] + p[5] for p in placed), default=0),
                 max((p[1] + p[4] for p in placed), default=0),
                 max((p[0] + p[3] for p in placed), default=0)))
    # dedupe preserving order
    seen = set()
    uniq = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    uniq.sort(key=lambda c: (c[0], c[1], c[2]))
    return uniq


def _place_in_bin(placed, W, H, D, w, h, d):
    cands = _candidate_positions(W, H, D, placed, w, h, d)
    for (z, y, x) in cands:
        if _fits_in_bin(x, y, z, w, h, d, W, H, D, placed):
            return (x, y, z)
    return None


def _score_placement(placement, boxes, container):
    W, H, D = container
    cont_vol = W * H * D
    if not placement:
        return float("inf")
    bins = {}
    for (b, x, y, z, w, h, d) in placement:
        bins.setdefault(b, []).append((w, h, d))
    B = len(bins)
    last = max(bins.keys())
    last_vol = sum(w * h * d for (w, h, d) in bins[last])
    frac = last_vol / cont_vol if cont_vol > 0 else 1.0
    return B + 0.1 * (1 - frac)


def pack(boxes, container):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    W, H, D = container

    # index boxes with their dims
    indexed = [(i, w, h, d, w * h * d) for (i, (w, h, d)) in enumerate(boxes)]
    # First-fit decreasing by volume, tie-break by largest dimension
    indexed.sort(key=lambda t: (-t[4], -(t[1] + t[2] + t[3])))

    bins = []  # list of lists of (x, y, z, w, h, d)
    box_bin = {}
    box_pos = {}
    box_dim = {}

    for (i, w, h, d, vol) in indexed:
        placed_any = False
        for bidx, placed in enumerate(bins):
            pos = _place_in_bin(placed, W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                placed.append((x, y, z, w, h, d))
                box_bin[i] = bidx
                box_pos[i] = (x, y, z)
                box_dim[i] = (w, h, d)
                placed_any = True
                break
        if not placed_any:
            # open new bin
            bins.append([(0, 0, 0, w, h, d)])
            box_bin[i] = len(bins) - 1
            box_pos[i] = (0, 0, 0)
            box_dim[i] = (w, h, d)
        if time.time() > deadline - 20.0:
            break

    # Build current placement
    def build_placement():
        out = []
        for i in range(len(boxes)):
            if i not in box_bin:
                # fallback: place in bin 0 at origin (shouldn't happen)
                w, h, d = boxes[i]
                out.append((0, 0, 0, w, h, d))
            else:
                b = box_bin[i]
                x, y, z = box_pos[i]
                w, h, d = box_dim[i]
                out.append((b, x, y, z, w, h, d))
        return out

    current = build_placement()
    best = current
    best_score = _score_placement(best, boxes, container)

    # Local improvement: try to move boxes from higher bins into lower bins
    # Iterate a few times, time-checked.
    n = len(boxes)
    for it in range(3):
        if time.time() > deadline - 15.0:
            break
        improved = False
        # process boxes in descending bin order
        order = sorted(range(n), key=lambda i: box_bin.get(i, 0), reverse=True)
        for i in order:
            if time.time() > deadline - 15.0:
                break
            b = box_bin[i]
            if b == 0:
                continue
            w, h, d = box_dim[i]
            # try to move into each lower bin
            for tb in range(b):
                placed = bins[tb]
                pos = _place_in_bin(placed, W, H, D, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, w, h, d))
                    box_bin[i] = tb
                    box_pos[i] = (x, y, z)
                    improved = True
                    break
        if not improved:
            break

    # Compact toward origin within each bin to free space (light)
    for bidx in range(len(bins)):
        if time.time() > deadline - 10.0:
            break
        placed = bins[bidx]
        # sort by (z, y, x) and try to shift each toward origin
        # simple: for each box, try decreasing x, y, z to origin
        for k in range(len(placed)):
            x, y, z, w, h, d = placed[k]
            others = [placed[j] for j in range(len(placed)) if j != k]
            new_x = x
            for nx in range(0, x + 1):
                if _fits_in_bin(nx, y, z, w, h, d, W, H, D, others):
                    new_x = nx
                    break
            new_y = y
            for ny in range(0, y + 1):
                if _fits_in_bin(new_x, ny, z, w, h, d, W, H, D, others):
                    new_y = ny
                    break
            new_z = z
            for nz in range(0, z + 1):
                if _fits_in_bin(new_x, new_y, nz, w, h, d, W, H, D, others):
                    new_z = nz
                    break
            if (new_x, new_y, new_z) != (x, y, z):
                placed[k] = (new_x, new_y, new_z, w, h, d)

    # Rebuild and re-score
    current = build_placement()
    cur_score = _score_placement(current, boxes, container)
    if cur_score < best_score:
        best = current
        best_score = cur_score

    # Final safety: ensure validity by a simple rebuild check; if invalid, fall back to a trivial valid packing
    def validate(placement):
        binsv = {}
        for (b, x, y, z, w, h, d) in placement:
            if b < 0:
                return False
            if x + w > W or y + h > H or z + d > D or x < 0 or y < 0 or z < 0:
                return False
            binsv.setdefault(b, []).append((x, y, z, w, h, d))
        for b, lst in binsv.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if _overlap(lst[i], lst[j]):
                        return False
        return True

    if not validate(best):
        # trivial valid: one box per bin
        best = [(i, 0, 0, 0, w, h, d) for (i, (w, h, d)) in enumerate(boxes)]

    return best

# EVOLVE-BLOCK-END
