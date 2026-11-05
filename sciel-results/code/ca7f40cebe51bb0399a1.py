"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _fits(box, pos, occupied, W, H, D):
    """Check if box at pos fits in container and doesn't overlap any occupied box."""
    bx, by, bz, bw, bh, bd = box
    px, py, pz = pos
    # Container bounds
    if px < 0 or py < 0 or pz < 0:
        return False
    if px + bw > W or py + bh > H or pz + bd > D:
        return False
    # Overlap check with all occupied boxes
    for ox, oy, oz, ow, oh, od in occupied:
        if not (px + bw <= ox or ox + ow <= px or
                py + bh <= oy or oy + oh <= py or
                pz + bd <= oz or oz + od <= pz):
            return False
    return True


def _find_position(box, occupied, W, H, D):
    """Find the best position for a box in a bin using a grid-based search.
    
    We search candidate positions sorted by (z, y, x) to prefer lower, earlier positions.
    This is a simplified best-fit approach: we check positions on a grid derived from
    existing box edges.
    """
    bx, by, bz, bw, bh, bd = box
    
    # Build candidate x, y, z coordinates from existing box edges
    x_coords = [0]
    y_coords = [0]
    z_coords = [0]
    for ox, oy, oz, ow, oh, od in occupied:
        if ox + ow <= W:
            x_coords.append(ox + ow)
        if oy + oh <= H:
            y_coords.append(oy + oh)
        if oz + od <= D:
            z_coords.append(oz + od)
    
    # Add container dimensions as upper bounds (we'll filter later)
    x_coords = sorted(set(x_coords))
    y_coords = sorted(set(y_coords))
    z_coords = sorted(set(z_coords))
    
    # Try all combinations, sorted by (z, y, x) for best-fit (lowest, earliest first)
    best_pos = None
    best_score = float('inf')
    
    for pz in z_coords:
        if pz + bd > D:
            break
        for py in y_coords:
            if py + bh > H:
                break
            for px in x_coords:
                if px + bw > W:
                    break
                # Check overlap
                valid = True
                for ox, oy, oz, ow, oh, od in occupied:
                    if not (px + bw <= ox or ox + ow <= px or
                            py + bh <= oy or oy + oh <= py or
                            pz + bd <= oz or oz + od <= pz):
                        valid = False
                        break
                if valid:
                    # Score: prefer lower z, then lower y, then lower x (minimize sum)
                    score = pz * 10000 + py * 100 + px
                    if score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
    
    return best_pos


def _find_position_fast(box, occupied, W, H, D):
    """Faster position finder using a limited candidate set."""
    bx, by, bz, bw, bh, bd = box
    
    # Candidate positions: (0,0,0) plus positions right/above/behind each existing box
    candidates = set()
    candidates.add((0, 0, 0))
    for ox, oy, oz, ow, oh, od in occupied:
        # Right of box
        if ox + ow + bw <= W:
            candidates.add((ox + ow, oy, oz))
        # Above box
        if oy + oh + bh <= H:
            candidates.add((ox, oy + oh, oz))
        # Behind box
        if oz + od + bd <= D:
            candidates.add((ox, oy, oz + od))
        # Combinations
        if ox + ow + bw <= W and oy + oh + bh <= H:
            candidates.add((ox + ow, oy + oh, oz))
        if ox + ow + bw <= W and oz + od + bd <= D:
            candidates.add((ox + ow, oy, oz + od))
        if oy + oh + bh <= H and oz + od + bd <= D:
            candidates.add((ox, oy + oh, oz + od))
    
    # Also try (0, 0, z) for various z, (0, y, 0) for various y, (x, 0, 0) for various x
    for ox, oy, oz, ow, oh, od in occupied:
        if oz + od + bd <= D:
            candidates.add((0, 0, oz + od))
        if oy + oh + bh <= H:
            candidates.add((0, oy + oh, 0))
        if ox + ow + bw <= W:
            candidates.add((ox + ow, 0, 0))
    
    best_pos = None
    best_score = float('inf')
    
    for px, py, pz in candidates:
        if px < 0 or py < 0 or pz < 0:
            continue
        if px + bw > W or py + bh > H or pz + bd > D:
            continue
        # Check overlap
        valid = True
        for ox, oy, oz, ow, oh, od in occupied:
            if not (px + bw <= ox or ox + ow <= px or
                    py + bh <= oy or oy + oh <= py or
                    pz + bd <= oz or oz + od <= pz):
                valid = False
                break
        if valid:
            # Score: prefer lower z, then lower y, then lower x
            score = pz * 10000 + py * 100 + px
            if score < best_score:
                best_score = score
                best_pos = (px, py, pz)
    
    return best_pos


def pack(boxes, container):
    """Pack boxes into bins using Best Fit Decreasing with multiple heuristics."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending for BFD
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Try multiple packing strategies and keep the best
    best_placements = None
    best_score = float('inf')
    
    # Strategy 1: BFD with best-fit position
    def strategy_bfd():
        bins = []  # list of lists of (x, y, z, w, h, d)
        placements = [None] * n
        
        for idx, w, h, d, vol in indexed_boxes:
            box = (w, h, d, vol)
            placed = False
            
            # Try existing bins first
            for b_idx in range(len(bins)):
                if time.time() > deadline:
                    return None  # timeout
                pos = _find_position_fast((w, h, d, 0, 0, 0), bins[b_idx], W, H, D)
                if pos is not None:
                    px, py, pz = pos
                    bins[b_idx].append((px, py, pz, w, h, d))
                    placements[idx] = (b_idx, px, py, pz, w, h, d)
                    placed = True
                    break
            
            # If not placed, try new bin
            if not placed:
                pos = _find_position_fast((w, h, d, 0, 0, 0), [], W, H, D)
                if pos is None:
                    # Box doesn't fit in empty container - should not happen
                    pos = (0, 0, 0)
                new_bin = [(pos[0], pos[1], pos[2], w, h, d)]
                bins.append(new_bin)
                placements[idx] = (len(bins) - 1, pos[0], pos[1], pos[2], w, h, d)
        
        if any(p is None for p in placements):
            return None
        return placements
    
    # Strategy 2: BFD with shelf-like placement (faster, may be less optimal)
    def strategy_shelf():
        bins = []
        placements = [None] * n
        
        for idx, w, h, d, vol in indexed_boxes:
            placed = False
            
            for b_idx in range(len(bins)):
                if time.time() > deadline:
                    return None
                # Shelf approach: try to fit in current shelf
                bin_boxes = bins[b_idx]
                # Find max z used
                max_z = max((bz + bd for _, _, bz, _, _, bd in bin_boxes), default=0)
                # Try placing at (0, 0, max_z)
                pos = (0, 0, max_z)
                if _fits((w, h, d, 0, 0, 0), pos, bin_boxes, W, H, D):
                    bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d))
                    placements[idx] = (b_idx, pos[0], pos[1], pos[2], w, h, d)
                    placed = True
                    break
                # Try other positions
                pos2 = _find_position_fast((w, h, d, 0, 0, 0), bin_boxes, W, H, D)
                if pos2 is not None:
                    bins[b_idx].append((pos2[0], pos2[1], pos2[2], w, h, d))
                    placements[idx] = (b_idx, pos2[0], pos2[1], pos2[2], w, h, d)
                    placed = True
                    break
            
            if not placed:
                pos = (0, 0, 0)
                if not _fits((w, h, d, 0, 0, 0), pos, [], W, H, D):
                    return None
                new_bin = [(pos[0], pos[1], pos[2], w, h, d)]
                bins.append(new_bin)
                placements[idx] = (len(bins) - 1, pos[0], pos[1], pos[2], w, h, d)
        
        if any(p is None for p in placements):
            return None
        return placements
    
    # Run strategies
    for strategy in [strategy_bfd, strategy_shelf]:
        if time.time() > deadline:
            break
        result = strategy()
        if result is not None:
            # Calculate score
            bin_labels = set(p[0] for p in result)
            B = len(bin_labels)
            if B == 0:
                score = 0
            else:
                max_bin = max(bin_labels)
                # Calculate occupied volume in last bin
                last_bin_vol = 0
                for p in result:
                    if p[0] == max_bin:
                        last_bin_vol += p[4] * p[5] * p[6]
                container_vol = W * H * D
                if container_vol > 0:
                    fraction = last_bin_vol / container_vol
                else:
                    fraction = 0
                score = B + 0.1 * (1 - fraction)
            
            if score < best_score:
                best_score = score
                best_placements = result
    
    # If no valid placement found, fall back to simple shelf
    if best_placements is None:
        # Simple shelf as fallback
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
        best_placements = placements
    
    return best_placements

# EVOLVE-BLOCK-END
