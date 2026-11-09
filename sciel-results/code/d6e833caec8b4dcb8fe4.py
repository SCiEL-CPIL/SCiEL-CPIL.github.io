"""3D Bin Packing via Maximal Empty Prism (MEP) spatial partitioning + local search."""
import time
import random

def pack(boxes, container):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    W, H, D = container
    container_vol = W * H * D

    n = len(boxes)
    if n == 0:
        return []

    # Each bin maintains a list of MEPs: (x, y, z, w, h, d)
    # A bin's free space is the union of its MEPs (they are non-overlapping by construction)
    bins = []  # list of lists of MEPs

    def mep_volume(m):
        return m[3] * m[4] * m[5]

    def mep_fits(m, w, h, d):
        return m[3] >= w and m[4] >= h and m[5] >= d

    def split_mep(m, x, y, z, w, h, d):
        """Split MEP m after placing box (x,y,z,w,h,d) inside it.
        Returns up to 6 new non-overlapping sub-rectangles."""
        mx, my, mz, mw, mh, md = m
        results = []
        # 1. Left slice (full height/depth)
        if x > mx:
            results.append((mx, my, mz, x - mx, mh, md))
        # 2. Right slice (full height/depth)
        if x + w < mx + mw:
            results.append((x + w, my, mz, mx + mw - x - w, mh, md))
        
        # The middle x-range is [x, x+w]
        # 3. Bottom slice (within middle x-range)
        if y > my:
            results.append((x, my, mz, w, y - my, md))
        # 4. Top slice (within middle x-range)
        if y + h < my + mh:
            results.append((x, y + h, mz, w, my + mh - y - h, md))
            
        # The middle y-range is [y, y+h]
        # 5. Front slice (within middle x and y ranges)
        if z > mz:
            results.append((x, y, mz, w, h, z - mz))
        # 6. Back slice (within middle x and y ranges)
        if z + d < mz + md:
            results.append((x, y, z + d, w, h, mz + md - z - d))
            
        # Filter out zero/negative dimension slivers
        return [r for r in results if r[3] > 0 and r[4] > 0 and r[5] > 0]

    def place_in_bin(bin_meps, w, h, d):
        """Try to place box in a bin represented by its MEP list.
        Returns (x, y, z) if successful, None otherwise.
        Best-fit: choose the MEP where the box fits and the remaining volume is minimized."""
        best = None
        best_remaining = float('inf')

        for i, m in enumerate(bin_meps):
            if not mep_fits(m, w, h, d):
                continue
            # Try to place at the origin of this MEP (bottom-left-front corner)
            x, y, z = m[0], m[1], m[2]
            remaining = mep_volume(m) - w * h * d
            if remaining < best_remaining:
                best_remaining = remaining
                best = (i, x, y, z)

        if best is None:
            return None

        idx, x, y, z = best
        m = bin_meps[idx]
        new_meps = split_mep(m, x, y, z, w, h, d)
        # Replace the original MEP with its splits
        bin_meps[idx:idx+1] = new_meps
        return (x, y, z)

    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed = sorted(enumerate(boxes), key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    placements = [None] * n

    for orig_idx, (w, h, d) in indexed:
        if time.time() > DEADLINE:
            break

        placed = False
        # Try existing bins first (best-fit: smallest remaining volume after placement)
        best_bin = -1
        best_pos = None
        best_remaining = float('inf')

        for bi, bin_meps in enumerate(bins):
            # Find the best MEP in this bin for this box
            for i, m in enumerate(bin_meps):
                if not mep_fits(m, w, h, d):
                    continue
                x, y, z = m[0], m[1], m[2]
                remaining = mep_volume(m) - w * h * d
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = bi
                    best_pos = (x, y, z)
                    best_mep_idx = i

        if best_bin != -1:
            # Execute the placement
            m = bins[best_bin][best_mep_idx]
            new_meps = split_mep(m, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            bins[best_bin][best_mep_idx:best_mep_idx+1] = new_meps
            placements[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        else:
            # Open a new bin
            bi = len(bins)
            initial_mep = (0, 0, 0, W, H, D)
            new_meps = split_mep(initial_mep, 0, 0, 0, w, h, d)
            bins.append(new_meps)
            placements[orig_idx] = (bi, 0, 0, 0, w, h, d)

    # Fill any unplaced boxes (shouldn't happen, but safety)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            bi = len(bins)
            initial_mep = (0, 0, 0, W, H, D)
            new_meps = split_mep(initial_mep, 0, 0, 0, w, h, d)
            bins.append(new_meps)
            placements[i] = (bi, 0, 0, 0, w, h, d)

    # Local search: try to move boxes from higher bins to lower bins to reduce B
    if time.time() < DEADLINE - 15:
        def rebuild_bin_meps(boxes_in_bin):
            """Rebuild the MEP list for a bin given its boxes."""
            meps = [(0, 0, 0, W, H, D)]
            for (bx, by, bz, bw, bh, bd) in boxes_in_bin:
                # Find the MEP that contains this box
                placed = False
                for i, m in enumerate(meps):
                    if (m[0] <= bx and bx + bw <= m[0] + m[3] and
                        m[1] <= by and by + bh <= m[1] + m[4] and
                        m[2] <= bz and bz + bd <= m[2] + m[5]):
                        # Replace m with splits
                        new_meps = split_mep(m, bx, by, bz, bw, bh, bd)
                        meps[i:i+1] = new_meps
                        placed = True
                        break
                if not placed:
                    # This shouldn't happen if boxes are valid
                    pass
            return meps

        max_bin = max(p[0] for p in placements) if placements else 0
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            # Try to move boxes from the highest bin to lower bins
            for i in range(n):
                bi, x, y, z, w, h, d = placements[i]
                if bi < max_bin:
                    continue
                # Try moving to each lower bin
                for target in range(bi - 1, -1, -1):
                    # Save state for rollback
                    old_bins_target = [m for m in bins[target]]
                    
                    # Try to place in target
                    pos = place_in_bin(bins[target], w, h, d)
                    if pos is not None:
                        # Remove from source bin
                        source_boxes = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placements if p[0] == bi and p is not placements[i]]
                        bins[bi] = rebuild_bin_meps(source_boxes)
                        
                        # Update placement
                        placements[i] = (target, pos[0], pos[1], pos[2], w, h, d)
                        
                        # If we moved a box out of the last bin, check if it's empty
                        if not any(p[0] == bi for p in placements):
                            # Bin is empty, we can potentially reduce max_bin
                            # But we don't need to delete it, just note it
                            pass
                            
                        improved = True
                        # Recalculate max_bin
                        max_bin = max(p[0] for p in placements) if placements else 0
                        break
                    else:
                        # Rollback target bin
                        bins[target] = old_bins_target
                if improved:
                    break

    return placements
