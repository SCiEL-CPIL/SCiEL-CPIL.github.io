# EVOLVE-BLOCK-START
import time
import itertools

def _orientations(w, h, d):
    """Return unique orientations (permutations) of a box."""
    seen = set()
    out = []
    for p in itertools.permutations((w, h, d)):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _overlap(px, py, pz, pw, ph, pd, x, y, z, w, h, d):
    """True if two axis-aligned boxes strictly overlap."""
    return not (px + pw <= x or x + w <= px or
                py + ph <= y or y + h <= py or
                pz + pd <= z or z + d <= pz)

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check bounds and no overlap with already placed boxes."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlap(px, py, pz, pw, ph, pd, x, y, z, w, h, d):
            return False
    return True

def _find_position(placed, box, W, H, D, deadline):
    """
    Find the lowest (z, then y, then x) valid position for a box
    with a chosen orientation. Returns (x,y,z,ow,oh,od) or None.
    Uses candidate x,y from existing box edges plus 0, and z from 0.
    """
    ow, oh, od = box
    # Candidate x positions: 0 and right edges of placed boxes
    cand_x = [0]
    cand_y = [0]
    for (px, py, pz, pw, ph, pd) in placed:
        cand_x.append(px + pw)
        cand_y.append(py + ph)
    # Also add a few interior candidates near edges for better packing
    # Sort and dedupe, keep within bounds
    cand_x = sorted(set(c for c in cand_x if c + ow <= W))
    cand_y = sorted(set(c for c in cand_y if c + oh <= H))
    if not cand_x or not cand_y:
        return None

    # Try z = 0 first (bottom layer), then z from placed box tops
    cand_z = [0]
    for (px, py, pz, pw, ph, pd) in placed:
        cand_z.append(pz + pd)
    cand_z = sorted(set(c for c in cand_z if c + od <= D))
    if not cand_z:
        return None

    # Search in order of z, then y, then x to find "lowest" position
    # But we want best packing; prefer positions that use space well.
    # We'll iterate z ascending, y ascending, x ascending and return first fit.
    for z in cand_z:
        for y in cand_y:
            for x in cand_x:
                if _fits(placed, x, y, z, ow, oh, od, W, H, D):
                    return (x, y, z, ow, oh, od)
        if time.time() > deadline:
            break
    return None

def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # safety margin before 120s limit
    n = len(boxes)

    # Precompute orientations for each box
    all_orientations = [_orientations(w, h, d) for (w, h, d) in boxes]

    # Sort boxes by volume descending (bigger boxes first for better packing)
    # Keep track of original indices
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * n  # (bin, x, y, z, w, h, d) per original box index
    filled_volume = 0

    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        best = None
        best_score = -1
        # Try all orientations; pick the one that fits in the "lowest" position
        # Score: prefer lower z, then lower y, then lower x (to keep bottom full)
        for ow, oh, od in all_orientations[idx]:
            pos = _find_position(placed, (ow, oh, od), W, H, D, deadline)
            if pos is None:
                continue
            x, y, z, _, _, _ = pos
            # Score: minimize z, then y, then x
            score = -(z * 10000 + y * 100 + x)
            if score > best_score:
                best_score = score
                best = (x, y, z, ow, oh, od)
            if time.time() > deadline:
                break
        if best is not None:
            x, y, z, ow, oh, od = best
            placed.append((x, y, z, ow, oh, od))
            result[idx] = (0, x, y, z, ow, oh, od)
            filled_volume += ow * oh * od
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any remaining None (in case deadline hit mid-loop)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)

    return result
# EVOLVE-BLOCK-END
