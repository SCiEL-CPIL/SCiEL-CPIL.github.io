import time
import random
import math
import sys

# Safety margin: 150s limit. Use 140s to be safe.
DEADLINE = time.time() + 140.0

def _fits(placements, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box fits at (cx, cy, cz) without overlapping existing placements."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements:
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True

def _get_extreme_points(placements, W, H, D):
    """
    Generate Extreme Points (EPs).
    An EP is a point in 3D space that is a corner of the "skyline" formed by existing boxes.
    Specifically, we generate points at:
    1. (0,0,0)
    2. For each box, the points (x+w, y, z), (x, y+h, z), (x, y, z+d)
       and their combinations (x+w, y+h, z), (x+w, y, z+d), (x, y+h, z+d), (x+w, y+h, z+d)
    We filter out points that are "dominated" or inside existing boxes to reduce candidates,
    though for 40 boxes, brute force checking all combinations is fast enough.
    """
    eps = set()
    eps.add((0, 0, 0))
    
    for (px, py, pz, pw, ph, pd) in placements:
        # Generate all 8 corners of the box relative to the box's start
        # But we only care about "outward" faces or corners that define new potential spots
        # Standard EP generation:
        # Right face: (px+pw, py, pz), (px+pw, py+ph, pz), (px+pw, py, pz+pd), (px+pw, py+ph, pz+pd)
        # Top face: (px, py+ph, pz), (px, py+ph, pz+pd)
        # Front face: (px, py, pz+pd)
        
        # Let's just add all 8 corners of every box and the origin.
        # This is O(N) points.
        for dx in [0, pw]:
            for dy in [0, ph]:
                for dz in [0, pd]:
                    eps.add((px+dx, py+dy, pz+dz))
    
    # Filter: A point is a valid EP if it is not strictly inside any existing box.
    # Also, it must be within container bounds.
    valid_eps = []
    for p in eps:
        x, y, z = p
        if x >= W or y >= H or z >= D:
            continue
        
        inside = False
        for (px, py, pz, pw, ph, pd) in placements:
            if px <= x < px + pw and py <= y < py + ph and pz <= z < pz + pd:
                inside = True
                break
        if not inside:
            valid_eps.append(p)
            
    return valid_eps

def _pack_extreme_point(boxes, container, order):
    """
    Pack boxes using the Extreme Point heuristic.
    For each box, try all EPs in all bins, pick the one that minimizes a score.
    Score: primarily minimize Z (height), secondarily minimize X, then Y.
    This tends to create flat, compact stacks.
    """
    W, H, D = container
    n = len(boxes)
    bins = [] # List of lists of (x, y, z, w, h, d)
    
    for idx in order:
        w, h, d = boxes[idx]
        
        if not bins:
            bins.append([(0, 0, 0, w, h, d)])
            continue
            
        best_bin_idx = -1
        best_pos = None
        best_score = (float('inf'), float('inf'), float('inf'))
        
        for bi in range(len(bins)):
            placements = bins[bi]
            eps = _get_extreme_points(placements, W, H, D)
            
            for (cx, cy, cz) in eps:
                if not _fits(placements, cx, cy, cz, w, h, d, W, H, D):
                    continue
                
                # Score: Minimize Z, then X, then Y.
                # This encourages filling the bin from bottom to top, left to right, back to front.
                score = (cz, cx, cy)
                
                if score < best_score:
                    best_score = score
                    best_bin_idx = bi
                    best_pos = (cx, cy, cz)
        
        if best_bin_idx == -1:
            # No fit in existing bins, create new bin
            bins.append([(0, 0, 0, w, h, d)])
        else:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
    return bins

def _score_solution(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)
    
    if B == 0:
        return 0.0
        
    last_bin = bins[-1]
    
    if not last_bin:
        return B
        
    last_vol = sum(w * h * d for (_, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    
    return B + 0.1 * (1.0 - fill_ratio)

def _local_search(bins, boxes, container, max_iterations=200):
    """
    Perform local search to improve the score.
    Moves:
    1. Move a box from bin A to bin B (if it fits).
    2. Swap a box from bin A with a box from bin B (if both fit in each other's spots).
    
    We prioritize moves that reduce B (number of bins) or increase last-bin fill.
    """
    W, H, D = container
    C = W * H * D
    current_score = _score_solution(bins, container)
    
    # Map box index to its current bin and position
    # bins[i] is a list of (x, y, z, w, h, d). We need to know which original box index it is.
    # The input `boxes` list is fixed. The `bins` returned from _pack_extreme_point doesn't track indices.
    # We need to pass indices through. Let's modify the packing to return indices.
    
    # Actually, let's reconstruct the mapping.
    # The `order` used in packing determines the sequence.
    # Let's assume we pass the `order` to local search as well, or better, 
    # let's make _pack_extreme_point return (bins, order_used).
    # And then we can map box dimensions to indices.
    
    # For simplicity in this local search, let's assume `bins` contains tuples (idx, x, y, z, w, h, d).
    # I will modify _pack_extreme_point to return this format.
    
    improved = True
    iters = 0
    
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        
        current_score = _score_solution(bins, container)
        
        # Try moving boxes
        # We iterate through all bins and all boxes in them
        for bi in range(len(bins)):
            for bi_idx in range(len(bins[bi])):
                # Get the box to move
                item = bins[bi][bi_idx]
                # item is (idx, x, y, z, w, h, d)
                idx, ox, oy, oz, w, h, d = item
                
                # Remove it temporarily
                bins[bi].pop(bi_idx)
                
                # Try to place it in other bins
                placed = False
                for bj in range(len(bins)):
                    if bj == bi:
                        continue
                    
                    placements = bins[bj]
                    eps = _get_extreme_points(placements, W, H, D)
                    
                    # Also try (0,0,0) just in case
                    candidates = [(0,0,0)] + eps
                    random.shuffle(candidates)
                    
                    for (cx, cy, cz) in candidates:
                        if _fits(placements, cx, cy, cz, w, h, d, W, H, D):
                            # Check if this move improves the score
                            # We need to simulate the new bins state to score it.
                            # Since B might change if we empty a bin, we need to be careful.
                            # If we move the last box out of a bin, B decreases.
                            # If we move into the last bin, fill increases.
                            
                            # Tentative placement
                            bins[bj].append((idx, cx, cy, cz, w, h, d))
                            
                            # Check if bin bi is now empty
                            # If so, remove it from the list? 
                            # The score function uses len(bins). If a bin is empty, it shouldn't count.
                            # Let's filter empty bins before scoring.
                            non_empty_bins = [b for b in bins if len(b) > 0]
                            # But we need to maintain the structure for future moves.
                            # Let's just score the non-empty bins.
                            
                            new_score = _score_solution(non_empty_bins, container)
                            
                            if new_score < current_score - 1e-9:
                                # Accept move
                                current_score = new_score
                                improved = True
                                placed = True
                                break
                            else:
                                # Reject move
                                bins[bj].pop()
                    
                    if placed:
                        break
                
                if not placed:
                    # Put it back where it was
                    bins[bi].insert(bi_idx, item)
                    # If the bin became empty during the check and we didn't place it elsewhere, 
                    # it's still empty. But we inserted it back, so it's not empty.
                    
        # Optional: Remove empty bins if they exist (shouldn't happen if we manage carefully)
        bins = [b for b in bins if len(b) > 0]
        
    return bins

def _pack_with_indices(boxes, container, order):
    """
    Wrapper for _pack_extreme_point that returns bins with indices.
    """
    W, H, D = container
    n = len(boxes)
    bins = []
    
    for idx in order:
        w, h, d = boxes[idx]
        
        if not bins:
            bins.append([(idx, 0, 0, 0, w, h, d)])
            continue
            
        best_bin_idx = -1
        best_pos = None
        best_score = (float('inf'), float('inf'), float('inf'))
        
        for bi in range(len(bins)):
            placements_geom = [(x, y, z, w_, h_, d_) for (_, x, y, z, w_, h_, d_) in bins[bi]]
            eps = _get_extreme_points(placements_geom, W, H, D)
            
            for (cx, cy, cz) in eps:
                if not _fits(placements_geom, cx, cy, cz, w, h, d, W, H, D):
                    continue
                
                score = (cz, cx, cy)
                
                if score < best_score:
                    best_score = score
                    best_bin_idx = bi
                    best_pos = (cx, cy, cz)
        
        if best_bin_idx == -1:
            bins.append([(idx, 0, 0, 0, w, h, d)])
        else:
            bins[best_bin_idx].append((idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
    return bins

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    t_start = time.time()
    rng = random.Random(42)
    
    # Generate several orderings
    idx = list(range(n))
    orderings = []
    
    # 1. Volume Descending
    orderings.append(sorted(idx, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    # 2. Max Dim Descending
    orderings.append(sorted(idx, key=lambda i: max(boxes[i]), reverse=True))
    # 3. Sum Dim Descending
    orderings.append(sorted(idx, key=lambda i: sum(boxes[i]), reverse=True))
    # 4. Random
    for _ in range(5):
        s = idx[:]
        rng.shuffle(s)
        orderings.append(s)
        
    best_bins = None
    best_score = float('inf')
    
    for order in orderings:
        if time.time() > DEADLINE - 20: # Leave time for local search
            break
            
        bins = _pack_with_indices(boxes, container, order)
        
        # Apply Local Search to this candidate
        # Local search can be expensive, so limit iterations or time
        ls_start = time.time()
        bins_ls = _local_search(bins, boxes, container, max_iterations=50)
        ls_time = time.time() - ls_start
        
        score = _score_solution(bins_ls, container)
        
        if score < best_score:
            best_score = score
            best_bins = bins_ls
            
    if best_bins is None:
        # Fallback
        best_bins = [[(i, 0, 0, 0, w, h, d)] for i, (w, h, d) in enumerate(boxes)]
        
    # Construct final placements list
    final_placement_map = {}
    for bi, bin_content in enumerate(best_bins):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            final_placement_map[idx] = (bi, x, y, z, w, h, d)
            
    result = []
    for i in range(n):
        if i in final_placement_map:
            result.append(final_placement_map[i])
        else:
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result
