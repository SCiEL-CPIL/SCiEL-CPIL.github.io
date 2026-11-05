"""Improved 3D bin packing with FDD, skyline positions, and refinement pass."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 108.0  # 12s safety margin
    
    # Sort boxes by volume descending (FDD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Bins will store list of (x, y, z, w, h, d)
    bins = []
    placement_map = {}  # original index -> (bin_idx, x, y, z)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a simple shelf pattern in new bins
            for remaining_orig_idx, (rw, rh, rd) in indexed_boxes:
                if remaining_orig_idx not in placement_map:
                    placed = False
                    for b_idx, bin_contents in enumerate(bins):
                        pos = _try_shelf_place(bin_contents, W, H, D, rw, rh, rd)
                        if pos:
                            x, y, z = pos
                            placement_map[remaining_orig_idx] = (b_idx, x, y, z)
                            bin_contents.append((x, y, z, rw, rh, rd))
                            placed = True
                            break
                    if not placed:
                        b_idx = len(bins)
                        bins.append([(0, 0, 0, rw, rh, rd)])
                        placement_map[remaining_orig_idx] = (b_idx, 0, 0, 0)
            break
        
        placed = False
        # Best-fit: try all bins, pick the one with best position score
        best_bin_idx = -1
        best_pos = None
        best_score = None
        
        for b_idx, bin_contents in enumerate(bins):
            pos = _find_position(bin_contents, W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                score = _position_score(x, y, z, w, h, d, W, H, D)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
                    best_bin_idx = b_idx
        
        if best_bin_idx >= 0:
            x, y, z = best_pos
            placement_map[orig_idx] = (best_bin_idx, x, y, z)
            bins[best_bin_idx].append((x, y, z, w, h, d))
        else:
            # Create new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[orig_idx] = (b_idx, 0, 0, 0)
    
    # Refinement pass: try to improve positions of boxes in each bin
    if time.time() < deadline - 5.0:
        _refine(bins, W, H, D, deadline)
        # Update placement_map from bins
        for b_idx, bin_contents in enumerate(bins):
            for (bx, by, bz, bw, bh, bd) in bin_contents:
                for orig_idx, (ow, oh, od) in indexed_boxes:
                    if placement_map.get(orig_idx) and placement_map[orig_idx][0] == b_idx:
                        px, py, pz = placement_map[orig_idx][1], placement_map[orig_idx][2], placement_map[orig_idx][3]
                        if (px, py, pz) == (bx, by, bz) and (ow, oh, od) == (bw, bh, bd):
                            break
    
    # Rebuild placement_map from final bin states
    placement_map = {}
    for b_idx, bin_contents in enumerate(bins):
        for (bx, by, bz, bw, bh, bd) in bin_contents:
            for orig_idx, (ow, oh, od) in indexed_boxes:
                if (ow, oh, od) == (bw, bh, bd):
                    if orig_idx not in placement_map:
                        placement_map[orig_idx] = (b_idx, bx, by, bz)
                        break
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement_map[i]
        w, h, d = boxes[i]
        result.append((b_idx, x, y, z, w, h, d))
    
    return result

def _position_score(x, y, z, w, h, d, W, H, D):
    """Score a position: prefer low z, then low y, then low x. Penalize floating."""
    # Primary: minimize z (height), then y, then x
    score = z * 1.0 + y * 0.5 + x * 0.1
    # Small penalty for not being at a "corner" (aligned with existing box edges)
    return score

def _find_position(bin_contents, W, H, D, w, h, d):
    """Find the best position for a box in a bin using skyline-based candidates."""
    if not bin_contents:
        return (0, 0, 0)
    
    best_pos = None
    best_score = None
    
    # Generate candidate positions using skyline approach
    candidates = _generate_candidates(bin_contents, W, H, D, w, h, d)
    
    for (x, y, z) in candidates:
        if _fits(bin_contents, x, y, z, w, h, d, W, H, D):
            score = _position_score(x, y, z, w, h, d, W, H, D)
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (x, y, z)
    
    return best_pos

def _generate_candidates(bin_contents, W, H, D, w, h, d):
    """Generate candidate positions using edge-aligned approach with pruning."""
    # Candidate x positions: 0 and right edges of existing boxes
    x_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        x_candidates.add(bx + bw)
    x_candidates = sorted(x for x in x_candidates if x <= W - w)
    
    # Candidate y positions: 0 and top edges of existing boxes
    y_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        y_candidates.add(by + bh)
    y_candidates = sorted(y for y in y_candidates if y <= H - h)
    
    # Candidate z positions: 0 and front edges of existing boxes
    z_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        z_candidates.add(bz + bd)
    z_candidates = sorted(z for z in z_candidates if z <= D - d)
    
    # Generate all combinations but limit to reasonable number
    candidates = []
    for z in z_candidates:
        for y in y_candidates:
            for x in x_candidates:
                candidates.append((x, y, z))
    
    # Sort by score to find good positions early
    candidates.sort(key=lambda pos: _position_score(pos[0], pos[1], pos[2], w, h, d, W, H, D))
    
    # Limit candidates to keep search fast
    return candidates[:200]

def _fits(bin_contents, x, y, z, w, h, d, W, H, D):
    """Check if a box at (x,y,z) with dims (w,h,d) fits without overlap."""
    if x + w > W or y + h > H or z + d > D:
        return False
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True

def _refine(bins, W, H, D, deadline):
    """Try to improve positions of boxes in each bin by moving them to better spots."""
    for b_idx in range(len(bins)):
        if time.time() > deadline:
            break
        bin_contents = bins[b_idx]
        n_boxes = len(bin_contents)
        
        # Try to move each box to a better position
        for i in range(n_boxes):
            if time.time() > deadline:
                break
            box = bin_contents[i]
            bx, by, bz, bw, bh, bd = box
            
            # Remove box temporarily
            other_boxes = bin_contents[:i] + bin_contents[i+1:]
            
            # Find better position
            new_pos = _find_position(other_boxes, W, H, D, bw, bh, bd)
            
            if new_pos is not None:
                new_score = _position_score(new_pos[0], new_pos[1], new_pos[2], bw, bh, bd, W, H, D)
                old_score = _position_score(bx, by, bz, bw, bh, bd, W, H, D)
                
                if new_score < old_score - 1e-9:
                    bin_contents[i] = (new_pos[0], new_pos[1], new_pos[2], bw, bh, bd)

def _try_shelf_place(bin_contents, W, H, D, w, h, d):
    """Simple shelf placement for fallback."""
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            return None
        if _fits(bin_contents, x, y, z, w, h, d, W, H, D):
            return (x, y, z)
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return None
# EVOLVE-BLOCK-END
