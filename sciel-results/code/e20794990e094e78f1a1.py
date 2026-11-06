"""Model-and-optimize 3D bin packing with deadline guard."""
import time
import random
import math
from typing import List, Tuple
from collections import defaultdict


class FreeSpace:
    """Represents an axis-aligned empty cuboid within a bin."""
    __slots__ = ['x', 'y', 'z', 'w', 'h', 'd']
    
    def __init__(self, x, y, z, w, h, d):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.d = d
    
    def contains(self, bx, by, bz, bw, bh, bd):
        """Check if box (bx,by,bz,bw,bh,bd) fits entirely within this free space."""
        return (self.x <= bx and self.x + self.w >= bx + bw and
                self.y <= by and self.y + self.h >= by + bh and
                self.z <= bz and self.z + self.d >= bz + bd)
    
    def intersects(self, bx, by, bz, bw, bh, bd):
        """Check if box (bx,by,bz,bw,bh,bd) overlaps with this free space."""
        return (bx < self.x + self.w and self.x < bx + bw and
                by < self.y + self.h and self.y < by + bh and
                bz < self.z + self.d and self.z < bz + bd)
    
    def split(self, bx, by, bz, bw, bh, bd):
        """
        Split this free space around box (bx,by,bz,bw,bh,bd).
        Returns list of new free spaces that don't overlap the box.
        """
        new_spaces = []
        
        # Split along x-axis
        if self.x < bx:
            new_spaces.append(FreeSpace(self.x, self.y, self.z, bx - self.x, self.h, self.d))
        if bx + bw < self.x + self.w:
            new_spaces.append(FreeSpace(bx + bw, self.y, self.z, self.x + self.w - bx - bw, self.h, self.d))
        
        # Split along y-axis (for the remaining x-range)
        x_start = max(self.x, bx)
        x_end = min(self.x + self.w, bx + bw)
        x_w = x_end - x_start
        if x_w > 0:
            if self.y < by:
                new_spaces.append(FreeSpace(x_start, self.y, self.z, x_w, by - self.y, self.d))
            if by + bh < self.y + self.h:
                new_spaces.append(FreeSpace(x_start, by + bh, self.z, x_w, self.y + self.h - by - bh, self.d))
        
        # Split along z-axis (for the remaining x,y range)
        y_start = max(self.y, by)
        y_end = min(self.y + self.h, by + bh)
        y_h = y_end - y_start
        if y_h > 0:
            if self.z < bz:
                new_spaces.append(FreeSpace(x_start, y_start, self.z, x_w, y_h, bz - self.z))
            if bz + bd < self.z + self.d:
                new_spaces.append(FreeSpace(x_start, y_start, bz + bd, x_w, y_h, self.z + self.d - bz - bd))
        
        return new_spaces


class BinState:
    """Maintains free spaces for a bin to enable fast placement checks."""
    __slots__ = ['free_spaces', 'items', 'W', 'H', 'D', 'filled_vol']
    
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.free_spaces = [FreeSpace(0, 0, 0, W, H, D)]
        self.items = []  # List of (x, y, z, w, h, d)
        self.filled_vol = 0
    
    def find_position(self, w, h, d):
        """
        Find a valid position (x, y, z) for box (w, h, d) in this bin.
        Returns None if no position found.
        """
        if w > self.W or h > self.H or d > self.D:
            return None
        
        # Sort free spaces by volume descending to find best fit
        # Prefer spaces where the box fits with minimal wasted space
        candidates = []
        for fs in self.free_spaces:
            if fs.w >= w and fs.h >= h and fs.d >= d:
                # Box can fit in this space
                # Try placing at bottom-left-front of the free space
                pos = (fs.x, fs.y, fs.z)
                # Calculate waste: how much extra space the box leaves
                waste = (fs.w - w) + (fs.h - h) + (fs.d - d)
                candidates.append((waste, pos, fs))
        
        if not candidates:
            return None
        
        # Sort by waste (less waste is better), then by position (closer to origin)
        candidates.sort(key=lambda c: (c[0], c[1][0] + c[1][1] + c[1][2]))
        
        return candidates[0][1]
    
    def place_box(self, x, y, z, w, h, d):
        """Place a box at (x,y,z) with dimensions (w,h,d), updating free spaces."""
        self.items.append((x, y, z, w, h, d))
        self.filled_vol += w * h * d
        
        new_free_spaces = []
        for fs in self.free_spaces:
            if fs.intersects(x, y, z, w, h, d):
                # Split this free space
                new_free_spaces.extend(fs.split(x, y, z, w, h, d))
            else:
                # Keep this free space
                new_free_spaces.append(fs)
        
        # Merge overlapping free spaces (optional optimization)
        self.free_spaces = new_free_spaces
    
    def remove_box(self, x, y, z, w, h, d):
        """Remove a box and rebuild free spaces from scratch."""
        self.items = [(ix, iy, iz, iw, ih, id) for (ix, iy, iz, iw, ih, id) in self.items
                      if not (ix == x and iy == y and iz == z and iw == w and ih == h and id == d)]
        self.filled_vol -= w * h * d
        
        # Rebuild free spaces from scratch
        self.free_spaces = [FreeSpace(0, 0, 0, self.W, self.H, self.D)]
        for (ix, iy, iz, iw, ih, id) in self.items:
            new_free_spaces = []
            for fs in self.free_spaces:
                if fs.intersects(ix, iy, iz, iw, ih, id):
                    new_free_spaces.extend(fs.split(ix, iy, iz, iw, ih, id))
                else:
                    new_free_spaces.append(fs)
            self.free_spaces = new_free_spaces


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 115.0

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D

    # Sort indices by volume descending
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)

    def bin_score(bin_state: BinState):
        """
        Score for a bin: lower is better for placing a new box.
        Prefers bins that are more filled.
        """
        if bin_state.filled_vol == 0:
            return float('inf')
        fill_frac = bin_state.filled_vol / container_vol
        # Penalize bins with many small free spaces (fragmentation)
        num_free = len(bin_state.free_spaces)
        frag_penalty = max(0, (num_free - 1) * 0.05)
        return -fill_frac + frag_penalty

    def pack_with_order(ordering):
        """Pack boxes in given order using free-space representation."""
        bins = {}  # bin_id -> BinState
        placements = [None] * n
        all_placed = True

        for box_idx in ordering:
            if time.time() > deadline:
                all_placed = False
                break

            w, h, d = boxes[box_idx]

            # Try existing bins: prefer bin with best score
            best_bin = None
            best_pos = None
            best_score_val = float('inf')

            bin_ids = list(bins.keys())
            # Sort bins by score (lower is better)
            bin_scores = {bid: bin_score(bins[bid]) for bid in bin_ids}
            bin_ids_sorted = sorted(bin_ids, key=lambda bid: bin_scores[bid])

            for bin_id in bin_ids_sorted:
                pos = bins[bin_id].find_position(w, h, d)
                if pos is not None:
                    # Prefer positions that minimize the new bounding box growth
                    waste = pos[0] + pos[1] + pos[2]
                    if waste < best_score_val:
                        best_score_val = waste
                        best_bin = bin_id
                        best_pos = pos

            if best_bin is not None:
                bins[best_bin].place_box(best_pos[0], best_pos[1], best_pos[2], w, h, d)
                placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            else:
                # Open new bin
                new_bin_id = max(bins.keys()) + 1 if bins else 0
                bins[new_bin_id] = BinState(W, H, D)
                bins[new_bin_id].place_box(0, 0, 0, w, h, d)
                placements[box_idx] = (new_bin_id, 0, 0, 0, w, h, d)

        if all_placed and all(p is not None for p in placements):
            return placements
        return None

    def local_search_polish(placements, W, H, D, max_iterations=200):
        """
        Local search: try to move boxes to other bins, reposition within bins,
        or swap boxes to reduce total bins used or improve last-bin fill.
        Uses free-space representation for fast checks.
        """
        if not placements:
            return placements

        # Build initial bin states from placements
        bin_boxes = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_boxes[b].append(i)
        
        bins = {}
        for b in bin_boxes:
            bins[b] = BinState(W, H, D)
            for i in bin_boxes[b]:
                _, x, y, z, w, h, d = placements[i]
                bins[b].place_box(x, y, z, w, h, d)

        current_score = compute_score(placements, W, H, D)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1

            # Try repositioning each box within its current bin
            for box_idx in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b, x, y, z, w, h, d = placements[box_idx]
                
                # Remove box from bin
                bins[b].remove_box(x, y, z, w, h, d)
                
                # Try to find a better position in the same bin
                pos = bins[b].find_position(w, h, d)
                if pos is not None and (pos[0], pos[1], pos[2]) != (x, y, z):
                    # Check if this position is better (closer to origin)
                    if pos[0] + pos[1] + pos[2] < x + y + z:
                        new_placements = list(placements)
                        new_placements[box_idx] = (b, pos[0], pos[1], pos[2], w, h, d)
                        new_score = compute_score(new_placements, W, H, D)
                        if new_score < current_score:
                            placements = new_placements
                            current_score = new_score
                            bins[b].place_box(pos[0], pos[1], pos[2], w, h, d)
                            improved = True
                            continue
                
                # Put box back in original position
                bins[b].place_box(x, y, z, w, h, d)

            # Try moving each box to another bin if it fits
            for box_idx in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b, x, y, z, w, h, d = placements[box_idx]

                # Try other bins
                other_bins = [bid for bid in bins if bid != b]
                for target_bin in other_bins:
                    # Check if box fits in target bin
                    pos = bins[target_bin].find_position(w, h, d)
                    if pos is not None:
                        # Tentatively move
                        new_placements = list(placements)
                        new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], w, h, d)
                        new_score = compute_score(new_placements, W, H, D)
                        if new_score < current_score:
                            placements = new_placements
                            current_score = new_score
                            # Update bin states
                            bins[b].remove_box(x, y, z, w, h, d)
                            bins[target_bin].place_box(pos[0], pos[1], pos[2], w, h, d)
                            # Update bin groupings
                            bin_boxes[b].remove(box_idx)
                            if not bin_boxes[b]:
                                del bin_boxes[b]
                            bin_boxes[target_bin].append(box_idx)
                            improved = True
                            break

            # Also try swapping pairs of boxes in different bins
            for box_idx1 in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b1, x1, y1, z1, w1, h1, d1 = placements[box_idx1]
                for box_idx2 in range(box_idx1 + 1, n):
                    if time.time() > deadline - 5:
                        break
                    b2, x2, y2, z2, w2, h2, d2 = placements[box_idx2]
                    if b1 == b2:
                        continue

                    # Try swapping: put box1 in bin2's position and box2 in bin1's position
                    # First remove both boxes
                    bins[b1].remove_box(x1, y1, z1, w1, h1, d1)
                    bins[b2].remove_box(x2, y2, z2, w2, h2, d2)
                    
                    # Check if box1 can go in bin2
                    pos1_in_b2 = bins[b2].find_position(w1, h1, d1)
                    if pos1_in_b2 is None:
                        # Restore
                        bins[b1].place_box(x1, y1, z1, w1, h1, d1)
                        bins[b2].place_box(x2, y2, z2, w2, h2, d2)
                        continue

                    # Check if box2 can go in bin1
                    pos2_in_b1 = bins[b1].find_position(w2, h2, d2)
                    if pos2_in_b1 is None:
                        # Restore
                        bins[b1].place_box(x1, y1, z1, w1, h1, d1)
                        bins[b2].place_box(x2, y2, z2, w2, h2, d2)
                        continue

                    # Tentatively swap
                    new_placements = list(placements)
                    new_placements[box_idx1] = (b2, pos1_in_b2[0], pos1_in_b2[1], pos1_in_b2[2], w1, h1, d1)
                    new_placements[box_idx2] = (b1, pos2_in_b1[0], pos2_in_b1[1], pos2_in_b1[2], w2, h2, d2)
                    new_score = compute_score(new_placements, W, H, D)
                    if new_score < current_score:
                        placements = new_placements
                        current_score = new_score
                        # Update bin states
                        bins[b2].place_box(pos1_in_b2[0], pos1_in_b2[1], pos1_in_b2[2], w1, h1, d1)
                        bins[b1].place_box(pos2_in_b1[0], pos2_in_b1[1], pos2_in_b1[2], w2, h2, d2)
                        # Update bin groupings
                        bin_boxes[b1].remove(box_idx1)
                        bin_boxes[b2].append(box_idx1)
                        bin_boxes[b2].remove(box_idx2)
                        bin_boxes[b1].append(box_idx2)
                        improved = True
                        break
                    else:
                        # Restore
                        bins[b1].place_box(x1, y1, z1, w1, h1, d1)
                        bins[b2].place_box(x2, y2, z2, w2, h2, d2)

        return placements

    # Main optimization: try multiple orderings
    best_result = None
    best_score = float('inf')

    # Generate diverse orderings
    orderings = []

    # 1. Volume descending
    orderings.append(indices_sorted)

    # 2. Volume ascending
    orderings.append(list(reversed(indices_sorted)))

    # 3. Original order
    orderings.append(indices)

    # 4. Max dimension descending
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))

    # 5. Surface area descending
    orderings.append(sorted(indices, key=lambda i: 2 * (boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2]), reverse=True))

    # 6. Aspect ratio based
    orderings.append(sorted(indices, key=lambda i: min(boxes[i][0], boxes[i][1], boxes[i][2]) / max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))

    # 7. Randomized volume descending (two variants)
    for seed in [42, 123]:
        random.seed(seed)
        shuffled = indices_sorted[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    # 8. Weighted by volume^0.5
    orderings.append(sorted(indices, key=lambda i: math.sqrt(box_vols[i]), reverse=True))

    # 9. Sort by minimum dimension
    orderings.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))

    # 10. Sort by maximum dimension
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))

    # 11. Sort by (w+h+d) descending - larger perimeter first
    orderings.append(sorted(indices, key=lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], reverse=True))

    # 12. Sort by min(w,h) descending - wider bases first
    orderings.append(sorted(indices, key=lambda i: min(boxes[i][0], boxes[i][1]), reverse=True))

    # Run all orderings and keep the best
    for ordering in orderings:
        if time.time() > deadline:
            break

        result = pack_with_order(ordering)
        if result is not None:
            score = compute_score(result, W, H, D)
            if score < best_score:
                best_score = score
                best_result = result

    # If no valid result found, fall back to simple shelf
    if best_result is None:
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements

    # Local search polish to improve packing
    if best_result and time.time() < deadline - 10:
        best_result = local_search_polish(best_result, W, H, D)
        best_score = compute_score(best_result, W, H, D)

    # Post-process: relabel bins to be consecutive starting from 0
    if best_result:
        bin_labels = sorted(set(p[0] for p in best_result))
        label_map = {old: new for new, old in enumerate(bin_labels)}
        best_result = [(label_map[p[0]], p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result]

    return best_result


def compute_score(placements, W, H, D):
    if not placements:
        return 0.0

    bins_used = set(p[0] for p in placements)
    B = len(bins_used)

    last_bin = max(bins_used)

    container_vol = W * H * D
    last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)

    score = B + 0.1 * (1 - last_bin_vol / container_vol)
    return score
