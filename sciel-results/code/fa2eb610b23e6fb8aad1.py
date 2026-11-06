# EVOLVE-BLOCK-START
import time
import itertools
import math

def pack(boxes, container):
    """
    3D bin packing: place every box unrotated without overlap into numbered bins.
    Strategy: Best-Fit Decreasing with enhanced corner-position search and multi-strategy.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    vol = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D

    def get_corners_enhanced(bin_data, w, h, d):
        """Generate candidate corner positions for a new box in this bin."""
        if not bin_data:
            return [(0, 0, 0)]

        corners = set()
        # Origin is always a candidate
        corners.add((0, 0, 0))

        for (bx, by, bz, bw, bh, bd) in bin_data:
            # All 8 corners of each placed box
            for dx in [0, bw]:
                for dy in [0, bh]:
                    for dz in [0, bd]:
                        corners.add((bx + dx, by + dy, bz + dz))
            
            # Additional positions: shift by the new box's dimensions
            # This helps find positions where the new box sits adjacent
            corners.add((bx + bw, by, bz))
            corners.add((bx, by + bh, bz))
            corners.add((bx, by, bz + bd))
            # Also try positions that align with the new box's footprint
            corners.add((bx + bw, by, bz))
            corners.add((bx, by + bh, bz))
            corners.add((bx, by, bz + bd))

        # Filter corners that are within container bounds
        valid_corners = []
        for (cx, cy, cz) in corners:
            if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                if cx + w <= W and cy + h <= H and cz + d <= D:
                    valid_corners.append((cx, cy, cz))

        return valid_corners

    def can_place(bin_data, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_data:
            # Check overlap
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def find_best_position(bin_data, w, h, d):
        """
        Find the best position for a box in this bin.
        'Best' means the position that packs most tightly (minimize wasted space).
        Returns (x, y, z) or None if no fit.
        """
        corners = get_corners_enhanced(bin_data, w, h, d)
        best_pos = None
        best_score = float('inf')

        for (cx, cy, cz) in corners:
            if can_place(bin_data, cx, cy, cz, w, h, d):
                # Score: prefer positions that minimize the bounding box expansion
                # Use a weighted sum to encourage tight packing in all dimensions
                # Prioritize filling gaps: prefer positions closer to origin
                score = cx * 1.0 + cy * 1.0 + cz * 1.0
                # Additional penalty for creating large gaps
                if best_pos is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)

        return best_pos

    def find_best_position_v2(bin_data, w, h, d):
        """
        Alternative scoring: prefer positions that minimize the maximum coordinate
        (pack toward origin in all dimensions simultaneously).
        """
        corners = get_corners_enhanced(bin_data, w, h, d)
        best_pos = None
        best_score = float('inf')

        for (cx, cy, cz) in corners:
            if can_place(bin_data, cx, cy, cz, w, h, d):
                # Minimize the maximum extent (tighter packing)
                score = max(cx, cy, cz)
                # Tie-break with sum
                if best_pos is None:
                    best_pos = (cx, cy, cz)
                    best_score = score
                elif score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                elif score == best_score:
                    # Tie-break: prefer lower sum
                    if cx + cy + cz < best_pos[0] + best_pos[1] + best_pos[2]:
                        best_pos = (cx, cy, cz)

        return best_pos

    def try_pack(order, position_func):
        """Try a packing strategy with given order and position function."""
        bins = []  # list of lists of (x, y, z, w, h, d)
        placement_map = [None] * n

        for idx in order:
            if time.time() > deadline:
                # Fallback: place remaining in new bins
                for j in range(idx, n):
                    if placement_map[j] is None:
                        b = len(bins)
                        bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
                        placement_map[j] = (b, 0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])
                break

            w, h, d = boxes[idx]

            # Try to fit in existing bins first
            placed = False
            for b_idx in range(len(bins)):
                pos = position_func(bins[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    placement_map[idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                # Open a new bin
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placement_map[idx] = (b_idx, 0, 0, 0, w, h, d)

        # Check if all boxes are placed
        if any(p is None for p in placement_map):
            # Fill in any missing
            for j in range(n):
                if placement_map[j] is None:
                    b = len(bins)
                    bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
                    placement_map[j] = (b, 0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])

        # Calculate score
        if not bins:
            return float('inf'), placement_map

        B = len(bins)
        # Last bin is the one with the largest label
        last_bin_idx = max(b for b, *_ in placement_map)
        last_bin_vol = sum(w * h * d for (_, _, _, w, h, d) in bins[last_bin_idx])
        last_bin_fraction = last_bin_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - last_bin_fraction)

        return score, placement_map

    # Generate multiple sorting strategies
    strategies = []

    # Strategy 1: Volume descending
    order1 = sorted(range(n), key=lambda i: -vol[i])
    strategies.append((order1, find_best_position))

    # Strategy 2: Volume descending, different position function
    strategies.append((order1, find_best_position_v2))

    # Strategy 3: Sort by max dimension descending
    max_dim = [max(w, h, d) for w, h, d in boxes]
    order3 = sorted(range(n), key=lambda i: -max_dim[i])
    strategies.append((order3, find_best_position))

    # Strategy 4: Sort by surface area descending
    surface = [2*(w*h + h*d + w*d) for w, h, d in boxes]
    order4 = sorted(range(n), key=lambda i: -surface[i])
    strategies.append((order4, find_best_position))

    # Strategy 5: Sort by volume then by max dimension
    order5 = sorted(range(n), key=lambda i: (-vol[i], -max_dim[i]))
    strategies.append((order5, find_best_position_v2))

    # Try each strategy and keep the best
    best_score = float('inf')
    best_placement = None

    for order, pos_func in strategies:
        if time.time() > deadline:
            break
        score, placement = try_pack(order, pos_func)
        if score < best_score:
            best_score = score
            best_placement = placement

    # If no strategy completed, use a simple fallback
    if best_placement is None:
        best_placement = [None] * n
        bins = []
        for i in range(n):
            b = len(bins)
            bins.append([(0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])])
            best_placement[i] = (b, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return best_placement
# EVOLVE-BLOCK-END
