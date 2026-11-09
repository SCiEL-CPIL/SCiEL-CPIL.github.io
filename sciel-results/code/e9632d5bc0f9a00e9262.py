"""Rewritten 3D bin packing: Extreme Point (EP) based best-fit decreasing with local search polish."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-t[1][0]*t[1][1]*t[1][2], -t[1][0], -t[1][1], -t[1][2]))
    order = [idx for idx, _ in indexed]

    start = time.time()
    DEADLINE = 115.0  # safety margin under 120s (evaluator kills at 150s)

    # Each bin: list of (x, y, z, w, h, d)
    bins = []  # list of lists of placements

    def overlaps(p, existing):
        """Check if new placement p overlaps any existing in the same bin."""
        px, py, pz, pw, ph, pd = p
        for ex, ey, ez, ew, eh, ed in existing:
            if (px < ex + ew and ex < px + pw and
                py < ey + eh and ey < py + ph and
                pz < ez + ed and ez < pz + pd):
                return True
        return False

    def fits(p):
        px, py, pz, pw, ph, pd = p
        return px >= 0 and py >= 0 and pz >= 0 and px + pw <= W and py + ph <= H and pz + pd <= D

    def get_extreme_points(existing, w, h, d):
        """Generate candidate (x, y, z) positions using extreme points.
        An extreme point is a corner that is 'supported' by existing boxes or bin walls.
        """
        if not existing:
            return [(0, 0, 0)]
        
        # Collect all unique x, y, z coordinates from box faces
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for ex, ey, ez, ew, eh, ed in existing:
            xs.add(ex)
            xs.add(ex + ew)
            ys.add(ey)
            ys.add(ey + eh)
            zs.add(ez)
            zs.add(ez + ed)
        
        # Filter coordinates that can accommodate the new box
        xs = sorted([x for x in xs if x + w <= W])
        ys = sorted([y for y in ys if y + h <= H])
        zs = sorted([z for z in zs if z + d <= D])
        
        if not xs or not ys or not zs:
            return []
        
        # Generate candidate positions from combinations
        # To avoid combinatorial explosion, we limit the number of candidates
        # by prioritizing positions closer to the origin
        cands = set()
        cands.add((xs[0], ys[0], zs[0]))  # Always try origin
        
        # Add positions near the "front" of existing boxes
        for ex, ey, ez, ew, eh, ed in existing:
            # Adjacent positions
            if ex + ew <= W:
                cands.add((ex + ew, ey, ez))
            if ey + eh <= H:
                cands.add((ex, ey + eh, ez))
            if ez + ed <= D:
                cands.add((ex, ey, ez + ed))
            if ex + ew <= W and ey + eh <= H:
                cands.add((ex + ew, ey + eh, ez))
            if ex + ew <= W and ez + ed <= D:
                cands.add((ex + ew, ey, ez + ed))
            if ey + eh <= H and ez + ed <= D:
                cands.add((ex, ey + eh, ez + ed))
            if ex + ew <= W and ey + eh <= H and ez + ed <= D:
                cands.add((ex + ew, ey + eh, ez + ed))
        
        # Also add some positions from the coordinate grid, but limit to avoid explosion
        # Sort by distance from origin to prioritize tighter fits
        all_cands = []
        for cx in xs:
            for cy in ys:
                for cz in zs:
                    all_cands.append((cx, cy, cz))
        
        # Sort by distance from origin
        all_cands.sort(key=lambda p: p[0] + p[1] + p[2])
        
        # Take top candidates (limited for performance)
        max_cands = 200
        for c in all_cands[:max_cands]:
            cands.add(c)
        
        # Filter to valid positions
        valid = []
        for (cx, cy, cz) in cands:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                valid.append((cx, cy, cz))
        return valid

    def placement_score(pos, w, h, d, existing):
        """Lower is better: prefer positions that minimize wasted space."""
        x, y, z = pos
        # Primary: minimize distance from origin (packs toward corner)
        score = x * 1.0 + y * 1.0 + z * 1.0
        # Secondary: prefer positions adjacent to existing boxes
        touches = 0
        for ex, ey, ez, ew, eh, ed in existing:
            if (ex + ew == x or ex == x + w or
                ey + eh == y or ey == y + h or
                ez + ed == z or ez == z + d):
                touches += 1
        score -= touches * 0.5
        return score

    def best_fit_bin(w, h, d):
        """Find the best (bin_index, position) for a box, or None."""
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        for bi, existing in enumerate(bins):
            cands = get_extreme_points(existing, w, h, d)
            for pos in cands:
                p = (pos[0], pos[1], pos[2], w, h, d)
                if not overlaps(p, existing):
                    s = placement_score(pos, w, h, d, existing)
                    if s < best_score:
                        best_score = s
                        best_bin = bi
                        best_pos = pos
        if best_bin >= 0:
            return best_bin, best_pos
        return None, None

    # Main loop: place each box
    placements_by_box = [None] * n

    for step in order:
        if time.time() - start > DEADLINE:
            break
        w, h, d = boxes[step]
        bi, pos = best_fit_bin(w, h, d)
        if bi is None:
            # Start a new bin
            bi = len(bins)
            bins.append([])
            pos = (0, 0, 0)
        bins[bi].append((pos[0], pos[1], pos[2], w, h, d))
        placements_by_box[step] = (bi, pos[0], pos[1], pos[2], w, h, d)

    # Post-processing: local search to improve last bin packing
    if time.time() - start < DEADLINE - 5 and bins:
        last_bin_idx = max(bi for bi in range(len(bins)) if bins[bi])
        last_bin = bins[last_bin_idx]
        if len(last_bin) > 1:
            improved = True
            max_iters = 5
            iters = 0
            while improved and iters < max_iters and time.time() - start < DEADLINE - 2:
                improved = False
                iters += 1
                for i in range(len(last_bin)):
                    if time.time() - start > DEADLINE - 2:
                        break
                    x, y, z, w, h, d = last_bin[i]
                    others = [last_bin[j] for j in range(len(last_bin)) if j != i]
                    cands = get_extreme_points(others, w, h, d)
                    best_pos = None
                    best_s = x + y + z
                    for pos in cands:
                        p = (pos[0], pos[1], pos[2], w, h, d)
                        if not overlaps(p, others):
                            s = placement_score(pos, w, h, d, others)
                            if s < best_s - 0.3:
                                best_s = s
                                best_pos = pos
                    if best_pos is not None:
                        last_bin[i] = (best_pos[0], best_pos[1], best_pos[2], w, h, d)
                        improved = True
                        # Update placements_by_box
                        for k in range(n):
                            if placements_by_box[k] is not None and placements_by_box[k][0] == last_bin_idx:
                                if (placements_by_box[k][4:7] == (w, h, d) and
                                    (placements_by_box[k][1:4] == (x, y, z))):
                                    placements_by_box[k] = (last_bin_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d)
                                    break

    # Fallback for any unplaced boxes
    for step in order:
        if placements_by_box[step] is not None:
            continue
        w, h, d = boxes[step]
        placed = False
        for bi in range(len(bins)):
            simple_cands = [(0, 0, 0)]
            for ex, ey, ez, ew, eh, ed in bins[bi]:
                if ex + ew <= W:
                    simple_cands.append((ex + ew, ey, ez))
                if ey + eh <= H:
                    simple_cands.append((ex, ey + eh, ez))
                if ez + ed <= D:
                    simple_cands.append((ex, ey, ez + ed))
            for pos in simple_cands:
                p = (pos[0], pos[1], pos[2], w, h, d)
                if not overlaps(p, bins[bi]):
                    bins[bi].append(p)
                    placements_by_box[step] = (bi, pos[0], pos[1], pos[2], w, h, d)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements_by_box[step] = (bi, 0, 0, 0, w, h, d)

    # Build result
    result = []
    for i in range(n):
        if placements_by_box[i] is None:
            w, h, d = boxes[i]
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            result.append((bi, 0, 0, 0, w, h, d))
        else:
            result.append(placements_by_box[i])
    return result
# EVOLVE-BLOCK-END
