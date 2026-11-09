"""Rewritten 3D bin packing with best-fit decreasing + time-guarded local search."""
import time
import random

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start = time.time()
    DEADLINE = 110.0  # safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort indices by volume descending (best-fit decreasing)
    order = sorted(range(n), key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

    # Represent a bin as a list of placed boxes: (x, y, z, w, h, d)
    # We'll maintain bins as lists of placed items.

    def overlaps(p1, p2):
        # p = (x, y, z, w, h, d)
        return not (p1[0] + p1[3] <= p2[0] or p2[0] + p2[3] <= p1[0] or
                    p1[1] + p1[4] <= p2[1] or p2[1] + p2[4] <= p1[1] or
                    p1[2] + p1[5] <= p2[2] or p2[2] + p2[5] <= p1[2])

    def can_place(bin_items, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_items:
            if overlaps((x, y, z, w, h, d), (px, py, pz, pw, ph, pd)):
                return False
        return True

    def bin_volume(bin_items):
        return sum(w * h * d for (_, _, _, w, h, d) in bin_items)

    def score_solution(bins):
        """Compute B + 0.1*(1 - last_bin_vol/container_vol)."""
        if not bins:
            return 0.0
        B = len(bins)
        # Last bin is the one with the largest label; since we use 0..B-1, last is bins[-1]
        last_vol = bin_volume(bins[-1])
        cont_vol = W * H * D
        if cont_vol == 0:
            return float(B)
        return B + 0.1 * (1.0 - last_vol / cont_vol)

    # Greedy best-fit decreasing: for each box (in sorted order), try to place in existing bin
    # at the position that minimizes the "wasted" space (best-fit by minimum remaining volume after placement).
    # Candidate positions: (0,0,0) and along each axis after existing boxes.

    def candidate_positions(bin_items, w, h, d):
        """Generate candidate (x,y,z) positions to try in a bin."""
        cands = {(0, 0, 0)}
        for (px, py, pz, pw, ph, pd) in bin_items:
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
        # Filter to valid positions
        valid = []
        for (x, y, z) in cands:
            if x + w <= W and y + h <= H and z + d <= D:
                valid.append((x, y, z))
        return valid

    def best_fit_position(bin_items, w, h, d):
        """Find the best (x,y,z) in bin_items for box (w,h,d) minimizing wasted space."""
        cands = candidate_positions(bin_items, w, h, d)
        best_pos = None
        best_waste = float('inf')
        for (x, y, z) in cands:
            if can_place(bin_items, x, y, z, w, h, d):
                # Wasted space heuristic: distance from origin (prefer compact packing)
                waste = x + y + z
                if waste < best_waste:
                    best_waste = waste
                    best_pos = (x, y, z)
        return best_pos

    # Initial greedy packing
    bins = []  # list of lists of (x, y, z, w, h, d)
    box_to_bin = [None] * n  # which bin index each box goes to

    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        # Try existing bins, prefer the one with least remaining volume after placement (best fit)
        best_bin = -1
        best_pos = None
        best_remaining = float('inf')
        for bi, bin_items in enumerate(bins):
            pos = best_fit_position(bin_items, w, h, d)
            if pos is not None:
                remaining = W * H * D - bin_volume(bin_items) - w * h * d
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = bi
                    best_pos = pos
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            box_to_bin[idx] = best_bin
        else:
            # New bin
            bins.append([(0, 0, 0, w, h, d)])
            box_to_bin[idx] = len(bins) - 1

    # Build placements from bins
    def make_placements(bins, box_to_bin):
        placements = [None] * n
        for bi, bin_items in enumerate(bins):
            for (x, y, z, w, h, d) in bin_items:
                # Find which original box this is
                # We need to track which box index each placement corresponds to
                pass
        return placements

    # Better: track placements directly
    def greedy_pack():
        bins = []
        placements = [None] * n
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            best_bin = -1
            best_pos = None
            best_remaining = float('inf')
            for bi, bin_items in enumerate(bins):
                pos = best_fit_position(bin_items, w, h, d)
                if pos is not None:
                    remaining = W * H * D - bin_volume(bin_items) - w * h * d
                    if remaining < best_remaining:
                        best_remaining = remaining
                        best_bin = bi
                        best_pos = pos
            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                placements[idx] = (best_bin, x, y, z, w, h, d)
            else:
                bins.append([(0, 0, 0, w, h, d)])
                placements[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
        return bins, placements

    bins, placements = greedy_pack()
    best_score = score_solution(bins)
    best_placements = list(placements)

    # Local search: try to move boxes from later bins to earlier bins to reduce B
    # or to fill the last bin more.
    def try_move_box(bins, placements, box_idx):
        """Try to move box_idx to a different bin or position. Returns True if moved."""
        old_bin = placements[box_idx][0]
        w, h, d = boxes[box_idx]
        # Remove from old bin
        old_x, old_y, old_z = placements[box_idx][1], placements[box_idx][2], placements[box_idx][3]
        bins[old_bin] = [item for item in bins[old_bin] if not (item[0] == old_x and item[1] == old_y and item[2] == old_z)]

        # Try to place in other bins
        for bi in range(len(bins)):
            if bi == old_bin:
                continue
            pos = best_fit_position(bins[bi], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[bi].append((x, y, z, w, h, d))
                placements[box_idx] = (bi, x, y, z, w, h, d)
                # Remove empty bins
                while bins and not bins[-1]:
                    bins.pop()
                # Reindex placements
                for i in range(n):
                    if placements[i] is not None and placements[i][0] >= len(bins):
                        # This shouldn't happen if we only pop empty last bins
                        pass
                return True
        # Put back in old bin
        bins[old_bin].append((old_x, old_y, old_z, w, h, d))
        return False

    # Simple local search: iterate through boxes and try to move them
    improved = True
    while improved and (time.time() - start) < DEADLINE:
        improved = False
        for idx in order:
            if (time.time() - start) > DEADLINE:
                break
            old_score = score_solution(bins)
            if try_move_box(bins, placements, idx):
                new_score = score_solution(bins)
                if new_score < old_score:
                    improved = True
                    best_score = new_score
                    best_placements = list(placements)
                else:
                    # Revert
                    pass  # try_move_box already reverted if it didn't help... actually it doesn't revert on worse score
                    # We need to handle this differently

    # The above local search is flawed. Let's do a cleaner version.
    # Reset and do a proper local search.
    bins, placements = greedy_pack()
    best_score = score_solution(bins)
    best_placements = list(placements)

    def get_score(bins):
        return score_solution(bins)

    # Local search: try moving each box to every other bin
    for _pass in range(5):
        if (time.time() - start) > DEADLINE:
            break
        improved = False
        for idx in order:
            if (time.time() - start) > DEADLINE:
                break
            old_bin = placements[idx][0]
            w, h, d = boxes[idx]
            old_x, old_y, old_z = placements[idx][1], placements[idx][2], placements[idx][3]

            # Remove from old bin
            bins[old_bin] = [item for item in bins[old_bin]
                            if not (item[0] == old_x and item[1] == old_y and item[2] == old_z)]

            best_new_score = get_score(bins) + w * h * d  # penalize unplaced
            best_new_bin = -1
            best_new_pos = None

            for bi in range(len(bins)):
                if bi == old_bin:
                    continue
                pos = best_fit_position(bins[bi], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    sc = get_score(bins)
                    bins[bi].pop()
                    if sc < best_new_score:
                        best_new_score = sc
                        best_new_bin = bi
                        best_new_pos = (x, y, z)

            if best_new_bin >= 0:
                x, y, z = best_new_pos
                bins[best_new_bin].append((x, y, z, w, h, d))
                placements[idx] = (best_new_bin, x, y, z, w, h, d)
                if best_new_score < best_score:
                    best_score = best_new_score
                    best_placements = list(placements)
                improved = True
            else:
                # Put back
                bins[old_bin].append((old_x, old_y, old_z, w, h, d))

        if not improved:
            break

    # Ensure all placements are valid and return
    result = list(best_placements)
    # Verify no None
    for i in range(n):
        if result[i] is None:
            # Should not happen, but fallback
            result[i] = (0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return result
# EVOLVE-BLOCK-END
