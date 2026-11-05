"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin before 120s limit
    container_vol = W * H * D
    
    # Pre-sort indices by volume descending (Best Fit Decreasing)
    # We try multiple sorting keys to see which works best
    keys = [
        lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2],  # volume
        lambda i: boxes[i][0]*boxes[i][1],              # base area
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # max dim
        lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][1],  # vol + area
    ]
    
    best_score = float('inf')
    best_placements = None
    
    for sort_key in keys:
        if time.time() > deadline:
            break
        
        order = sorted(range(n), key=sort_key, reverse=True)
        
        # Try both the sorted order and a few heuristic orderings
        candidate_orders = [order]
        
        # Add an ordering that prioritizes boxes that fit tightly
        # (smaller boxes might be better placed later, so try reverse of volume for small)
        # Actually, let's just try the primary sort and a "max dimension" sort
        
        for o in candidate_orders:
            if time.time() > deadline:
                break
            
            placements = _try_placement(boxes, container, o, deadline)
            if placements is None:
                continue
            
            # Try to consolidate bins to improve last-bin fill
            placements = _consolidate_bins(placements, boxes, container, deadline)
            
            score = _score(placements, boxes, container)
            if score < best_score:
                best_score = score
                best_placements = placements
    
    if best_placements is None:
        # Fallback: simple shelf packing
        return _shelf_pack(boxes, container)
    
    return best_placements


def _try_placement(boxes, container, order, deadline):
    """
    Attempt to place boxes in the given order using a Best Fit Decreasing style heuristic.
    For each box, try to find the best position in the first bin that has room,
    or open a new bin.
    
    Returns list of (bin, x, y, z, w, h, d) or None if failed.
    """
    W, H, D = container
    n = len(boxes)
    
    # Bin state: list of lists of placed boxes (x, y, z, w, h, d)
    bins = []  # each bin is a list of placed box tuples
    bin_volumes = []  # occupied volume per bin
    
    # For efficiency, we maintain a list of "candidate positions" per bin.
    # A candidate position is (x, y, z). We generate these from placed boxes.
    # Initially, each bin has one candidate: (0, 0, 0).
    
    # To keep it fast, we limit the number of candidate positions per bin.
    MAX_CANDIDATES = 300
    
    # placements: map from original index to (bin, x, y, z)
    result = [None] * n
    
    for idx in order:
        w, h, d = boxes[idx]
        
        # Check if box fits in container at all
        if w > W or h > H or d > D:
            return None  # Can't place this box
        
        best_bin = -1
        best_pos = None
        best_score_val = float('inf')
        
        num_bins = len(bins)
        
        # Try to find the best position across all existing bins
        # "Best" is defined by minimizing the resulting "waste" or by some heuristic.
        # For simplicity, we use a "First Fit Decreasing" variant with a scoring function.
        # Score for placing at (x,y,z) in bin b:
        #   - Prefer bins with higher occupied volume (to consolidate)
        #   - Prefer positions with smaller x, y, z (to pack tight)
        #   - Prefer positions that create less "dead space"
        
        for b in range(num_bins):
            if time.time() > deadline:
                return None
            
            # Get candidate positions for this bin
            candidates = _get_candidates(bins[b], W, H, D, MAX_CANDIDATES)
            
            for (cx, cy, cz) in candidates:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                
                # Check overlap with all boxes in this bin
                if _has_overlap(bins[b], cx, cy, cz, w, h, d):
                    continue
                
                # Calculate a score for this placement
                # Heuristic: prefer lower z, then lower y, then lower x
                # Also prefer bins that are already more full
                # Use a weighted combination that favors tight packing
                pos_score = cz * 10000 + cy * 1000 + cx
                bin_fill = bin_volumes[b] / (W * H * D) if W * H * D > 0 else 0
                # We want to minimize pos_score but maximize bin_fill
                # Combine: score = pos_score - bin_fill * 50000 (so full bins are strongly preferred)
                combined_score = pos_score - bin_fill * 50000
                
                if combined_score < best_score_val:
                    best_score_val = combined_score
                    best_bin = b
                    best_pos = (cx, cy, cz)
        
        # If no position found in existing bins, or if it's the first box, open a new bin
        if best_bin == -1:
            # Open a new bin
            best_bin = num_bins
            best_pos = (0, 0, 0)
            bins.append([])
            bin_volumes.append(0)
        
        # Place the box
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        bin_volumes[best_bin] += w * h * d
        result[idx] = (best_bin, x, y, z, w, h, d)
    
    # Convert to final format
    placements = []
    for i in range(n):
        b, x, y, z, w, h, d = result[i]
        placements.append((b, x, y, z, w, h, d))
    
    return placements


def _get_candidates(bins, W, H, D, max_candidates):
    """
    Generate candidate positions for a bin based on already placed boxes.
    Candidates are:
    - (0, 0, 0)
    - For each placed box, the positions just to the right, above, and in front
    - Also include positions at the boundaries of existing boxes
    """
    candidates = [(0, 0, 0)]
    seen = set()
    seen.add((0, 0, 0))
    
    # Collect all unique x, y, z coordinates from box edges
    x_coords = set([0, W])
    y_coords = set([0, H])
    z_coords = set([0, D])
    
    for (bx, by, bz, bw, bh, bd) in bins:
        x_coords.add(bx)
        x_coords.add(bx + bw)
        y_coords.add(by)
        y_coords.add(by + bh)
        z_coords.add(bz)
        z_coords.add(bz + bd)
    
    # Generate candidate positions from combinations of edge coordinates
    # This creates a grid of potential positions
    for cx in x_coords:
        if cx >= W:
            continue
        for cy in y_coords:
            if cy >= H:
                continue
            for cz in z_coords:
                if cz >= D:
                    continue
                pos = (cx, cy, cz)
                if pos not in seen:
                    candidates.append(pos)
                    seen.add(pos)
    
    # Also add the "adjacent" positions from each box (right, above, front)
    for (bx, by, bz, bw, bh, bd) in bins:
        # Right
        pos = (bx + bw, by, bz)
        if pos not in seen and pos[0] < W:
            candidates.append(pos)
            seen.add(pos)
        # Above
        pos = (bx, by + bh, bz)
        if pos not in seen and pos[1] < H:
            candidates.append(pos)
            seen.add(pos)
        # In front
        pos = (bx, by, bz + bd)
        if pos not in seen and pos[2] < D:
            candidates.append(pos)
            seen.add(pos)
        # Combinations
        pos = (bx + bw, by + bh, bz)
        if pos not in seen and pos[0] < W and pos[1] < H:
            candidates.append(pos)
            seen.add(pos)
        pos = (bx + bw, by, bz + bd)
        if pos not in seen and pos[0] < W and pos[2] < D:
            candidates.append(pos)
            seen.add(pos)
        pos = (bx, by + bh, bz + bd)
        if pos not in seen and pos[1] < H and pos[2] < D:
            candidates.append(pos)
            seen.add(pos)
        pos = (bx + bw, by + bh, bz + bd)
        if pos not in seen and pos[0] < W and pos[1] < H and pos[2] < D:
            candidates.append(pos)
            seen.add(pos)
    
    # Limit candidates
    if len(candidates) > max_candidates:
        # Sort by z, then y, then x to keep the "tightest" ones
        candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        candidates = candidates[:max_candidates]
    
    return candidates


def _has_overlap(bins, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any box in bins."""
    for (bx, by, bz, bw, bh, bd) in bins:
        # Overlap if intervals intersect in all 3 dimensions
        if (x < bx + bw and x + w > bx and
            y < by + bh and y + h > by and
            z < bz + bd and z + d > bz):
            return True
    return False


def _score(placements, boxes, container):
    """Calculate the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    W, H, D = container
    container_vol = W * H * D
    
    if not placements:
        return 0.0
    
    # Find max bin label
    max_bin = max(p[0] for p in placements)
    
    # Count occupied bins
    occupied = set(p[0] for p in placements)
    B = len(occupied)
    
    # Last bin occupied volume
    last_bin_vol = 0
    for p in placements:
        if p[0] == max_bin:
            _, _, _, _, w, h, d = p
            last_bin_vol += w * h * d
    
    last_frac = last_bin_vol / container_vol if container_vol > 0 else 0
    score = B + 0.1 * (1 - last_frac)
    
    return score


def _consolidate_bins(placements, boxes, container, deadline):
    """
    Try to consolidate bins by moving boxes from non-last bins to the last bin
    to improve the last-bin fill ratio without increasing B.
    """
    W, H, D = container
    if not placements:
        return placements
    
    # Group placements by bin
    from collections import defaultdict
    bin_boxes = defaultdict(list)
    for i, p in enumerate(placements):
        bin_boxes[p[0]].append(i)
    
    bins = sorted(bin_boxes.keys())
    if len(bins) <= 1:
        return placements
    
    last_bin = bins[-1]
    other_bins = bins[:-1]
    
    # Create a mutable copy of placements
    new_placements = list(placements)
    
    # Try to move boxes from other bins to the last bin
    # Sort other bins by their fill ratio (try to empty the emptiest ones first)
    bin_vols = {}
    for b in bins:
        vol = 0
        for idx in bin_boxes[b]:
            w, h, d = boxes[idx]
            vol += w * h * d
        bin_vols[b] = vol
    
    # Try to move boxes from non-last bins to last bin
    # For each box in non-last bins (sorted by volume descending), try to place it in last bin
    moved_count = 0
    for b in other_bins:
        if time.time() > deadline:
            break
        
        # Get boxes in this bin, sorted by volume descending
        box_indices = sorted(bin_boxes[b], key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in box_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            old_pos = new_placements[idx]
            
            # Try to find a position in the last bin
            last_bin_boxes = [(new_placements[i][1], new_placements[i][2], new_placements[i][3],
                              new_placements[i][4], new_placements[i][5], new_placements[i][6])
                              for i in bin_boxes[last_bin]]
            
            candidates = _get_candidates(last_bin_boxes, W, H, D, 200)
            found = False
            for (cx, cy, cz) in candidates:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                if _has_overlap(last_bin_boxes, cx, cy, cz, w, h, d):
                    continue
                
                # Found a valid position
                new_placements[idx] = (last_bin, cx, cy, cz, w, h, d)
                bin_boxes[last_bin].append(idx)
                bin_boxes[b].remove(idx)
                found = True
                moved_count += 1
                break
            
            if found:
                # Update last_bin_boxes for subsequent checks
                last_bin_boxes = [(new_placements[i][1], new_placements[i][2], new_placements[i][3],
                                  new_placements[i][4], new_placements[i][5], new_placements[i][6])
                                  for i in bin_boxes[last_bin]]
    
    return new_placements


def _shelf_pack(boxes, container):
    """Simple shelf packing as fallback."""
    W, H, D = container
    placements = []
    b = 0
    x, y, z = 0, 0, 0
    row_h, layer_d = 0, 0
    
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
    
    return placements


def _has_overlap(bins, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any box in bins."""
    for (bx, by, bz, bw, bh, bd) in bins:
        # Overlap if intervals intersect in all 3 dimensions
        if (x < bx + bw and x + w > bx and
            y < by + bh and y + h > by and
            z < bz + bd and z + d > bz):
            return True
    return False
# EVOLVE-BLOCK-END
