"""Rewritten 3D bin packing: best-fit decreasing with improved corner-candidate placement search."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (stable tie-break by dimensions)
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-t[1][0]*t[1][1]*t[1][2], -t[1][0], -t[1][1], -t[1][2]))
    order = [idx for idx, _ in indexed]

    # Deadline guard
    start = time.time()
    DEADLINE = 110.0  # safety margin under 120s

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

    def candidate_positions(existing, w, h, d):
        """Generate candidate (x, y, z) positions for a box of size (w,h,d) given existing placements.
        Uses a more comprehensive set of corner candidates including combinations of
        existing box edges to find tighter fits.
        """
        cands = set()
        cands.add((0, 0, 0))
        
        # Collect all unique x, y, z coordinates from existing boxes
        xs = set([0, W])
        ys = set([0, H])
        zs = set([0, D])
        for ex, ey, ez, ew, eh, ed in existing:
            xs.add(ex)
            xs.add(ex + ew)
            ys.add(ey)
            ys.add(ey + eh)
            zs.add(ez)
            zs.add(ez + ed)
        
        # Generate candidates from combinations of these coordinates
        # Limit to avoid combinatorial explosion: use top candidates near origin
        xs_sorted = sorted([x for x in xs if x + w <= W])
        ys_sorted = sorted([y for y in ys if y + h <= H])
        zs_sorted = sorted([z for z in zs if z + d <= D])
        
        # Generate a limited set of candidate positions
        # Strategy: try positions where each coordinate is either 0, or aligned with an existing box edge
        for cx in xs_sorted:
            for cy in ys_sorted:
                for cz in zs_sorted:
                    cands.add((cx, cy, cz))
        
        # Also add the original "adjacent" candidates for safety
        for ex, ey, ez, ew, eh, ed in existing:
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
        
        # Filter to valid positions
        valid = []
        for (cx, cy, cz) in cands:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                valid.append((cx, cy, cz))
        return valid

    def placement_score(pos, w, h, d, existing):
        """Lower is better: prefer positions that minimize wasted space.
        Use a weighted sum that penalizes gaps in x, y, z directions.
        """
        x, y, z = pos
        # Primary: minimize distance from origin (packs toward corner)
        score = x * 1.0 + y * 1.0 + z * 1.0
        # Secondary: prefer positions adjacent to existing boxes (reduces fragmentation)
        # Count how many existing boxes are "touching" this position
        touches = 0
        for ex, ey, ez, ew, eh, ed in existing:
            if (ex + ew == x or ex == x + w or
                ey + eh == y or ey == y + h or
                ez + ed == z or ez == z + d):
                touches += 1
        # Reward adjacency (subtract a small amount per touching box)
        score -= touches * 0.1
        return score

    def best_fit_bin(box_idx, w, h, d):
        """Find the best (bin_index, position) for a box, or None.
        Prioritizes bins that already have boxes (to minimize bin count).
        """
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        for bi, existing in enumerate(bins):
            cands = candidate_positions(existing, w, h, d)
            for pos in cands:
                p = (pos[0], pos[1], pos[2], w, h, d)
                if not overlaps(p, existing):
                    s = placement_score(pos, w, h, d, existing)
                    # Add a small penalty for using a new bin vs existing
                    # This encourages filling existing bins first
                    if len(existing) == 0:
                        s += 1000  # Strong penalty for empty bin (shouldn't happen in this loop)
                    if s < best_score:
                        best_score = s
                        best_bin = bi
                        best_pos = pos
        if best_bin >= 0:
            return best_bin, best_pos
        return None, None

    def new_bin_position(w, h, d):
        """Position for a new bin (always 0,0,0)."""
        return (0, 0, 0)

    # Main loop: place each box
    placements_by_box = [None] * n  # store (bin, x, y, z, w, h, d) per original index

    for step in order:
        if time.time() - start > DEADLINE:
            break
        w, h, d = boxes[step]
        bi, pos = best_fit_bin(step, w, h, d)
        if bi is None:
            # Start a new bin
            bi = len(bins)
            bins.append([])
            pos = new_bin_position(w, h, d)
        bins[bi].append((pos[0], pos[1], pos[2], w, h, d))
        placements_by_box[step] = (bi, pos[0], pos[1], pos[2], w, h, d)
    
    # Post-processing: try to improve last bin packing by repositioning boxes
    # This can help with the 0.1*(1 - last_bin_fraction) term
    if time.time() - start < DEADLINE - 5:  # Leave 5s buffer
        # Identify the last bin (highest label)
        if bins:
            last_bin_idx = max(bi for bi in range(len(bins)) if bins[bi])
            last_bin = bins[last_bin_idx]
            if len(last_bin) > 1:
                # Try to rearrange boxes in last bin for better packing
                # Simple local search: try moving each box to a different valid position
                improved = True
                max_iters = 3
                iters = 0
                while improved and iters < max_iters and time.time() - start < DEADLINE - 2:
                    improved = False
                    iters += 1
                    for i in range(len(last_bin)):
                        if time.time() - start > DEADLINE - 2:
                            break
                        x, y, z, w, h, d = last_bin[i]
                        others = [last_bin[j] for j in range(len(last_bin)) if j != i]
                        cands = candidate_positions(others, w, h, d)
                        best_pos = None
                        best_s = x + y + z  # current score
                        for pos in cands:
                            p = (pos[0], pos[1], pos[2], w, h, d)
                            if not overlaps(p, others):
                                s = placement_score(pos, w, h, d, others)
                                if s < best_s - 0.5:  # Require meaningful improvement
                                    best_s = s
                                    best_pos = pos
                        if best_pos is not None:
                            last_bin[i] = (best_pos[0], best_pos[1], best_pos[2], w, h, d)
                            improved = True
                            # Update placements_by_box for boxes in this bin
                            for bi in range(len(bins)):
                                for j, (px, py, pz, pw, ph, pd) in enumerate(bins[bi]):
                                    if (px, py, pz, pw, ph, pd) == last_bin[i]:
                                        # Find which original box this is
                                        for k in range(n):
                                            if placements_by_box[k] is not None and \
                                               placements_by_box[k][0] == bi and \
                                               placements_by_box[k][1:4] == (px, py, pz) and \
                                               placements_by_box[k][4:7] == (pw, ph, pd):
                                                placements_by_box[k] = (bi, px, py, pz, pw, ph, pd)
                                                break

    # If we timed out, try to place remaining boxes greedily into existing or new bins
    # Use a simpler fallback that's faster
    for step in order:
        if placements_by_box[step] is not None:
            continue
        w, h, d = boxes[step]
        placed = False
        for bi in range(len(bins)):
            # Try simple positions first for speed
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
            # Fallback: put in a new bin at origin (shouldn't happen)
            w, h, d = boxes[i]
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            result.append((bi, 0, 0, 0, w, h, d))
        else:
            result.append(placements_by_box[i])
    return result
# EVOLVE-BLOCK-END
