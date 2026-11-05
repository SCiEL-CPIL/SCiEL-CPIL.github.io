"""Targeted Volume Filling: pack until bin is nearly full, then force new bin to maximize last-bin density."""
import time
import random
import math

DEADLINE = time.time() + 145.0


def _fits(placements, cx, cy, cz, w, h, d, W, H, D):
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


def _get_anchors(placements, W, H, D):
    """Generate candidate positions: origin and faces of existing boxes."""
    anchors = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placements:
        # Check if anchor is within bounds
        if px + pw < W:
            anchors.append((px + pw, py, pz))
        if py + ph < H:
            anchors.append((px, py + ph, pz))
        if pz + pd < D:
            anchors.append((px, py, pz + pd))
    return anchors


def _pack_with_target(boxes, container, order, target_fill_ratio=0.95, max_probes=50):
    W, H, D = container
    container_vol = W * H * D
    bins = []  # list of (placements, volume)
    
    for idx in order:
        w, h, d = boxes[idx]
        vol = w * h * d
        
        # If box doesn't fit in empty bin, it can't be placed (shouldn't happen in valid input)
        if w > W or h > H or d > D:
            # Fallback: put in a new bin? No, task implies valid input. 
            # But to be safe, if it doesn't fit, we can't pack it. 
            # However, the problem states "places every box". 
            # Assuming all boxes fit in a single empty container.
            pass
            
        placed = False
        target_vol = target_fill_ratio * container_vol
        
        # Try to place in existing bins, but ONLY if the bin is below target fill
        # OR if this is the last box (to ensure it's placed)
        # Actually, we want to fill the CURRENT bin (last one) up to target.
        # If the last bin is below target, try to put it there.
        # If the last bin is above target, we should open a new bin.
        
        # Strategy: 
        # 1. If there are no bins, create one.
        # 2. If the last bin's volume + current box vol <= target_vol, try to place in last bin.
        # 3. If last bin is "full" (vol + box > target_vol), create a new bin.
        # 4. Exception: If this is the very last box, we MUST place it. 
        #    If last bin is full, we create a new bin for it. 
        #    This ensures the "last bin" in the output is the one containing the last box.
        #    Wait, the score uses the bin with the LARGEST LABEL. 
        #    If we create a new bin for the last box, that new bin is the "last bin".
        #    If that new bin only contains the last box, its fill ratio is low.
        #    So, we should try to fit the last box into the PREVIOUS bin if possible, 
        #    even if it exceeds the target, to keep the last bin dense?
        #    No, if we put it in a new bin, that new bin is the last one.
        #    If we put it in the previous bin, the previous bin is the last one.
        #    So, for the last box, we should prefer the bin that results in the best score.
        #    Actually, simpler: just follow the target rule. If target is reached, open new bin.
        #    If the last box opens a new bin, that bin has 1 box. 
        #    To avoid this, we can set target lower, or try to fit last box into existing bins if it fits.
        
        # Let's refine: 
        # For all boxes except the last one:
        #   If last bin vol + box vol <= target_vol: try to place in last bin.
        #   Else: create new bin.
        # For the last box:
        #   Try to place in the LAST bin (even if it exceeds target). 
        #   If it fits, place it there. 
        #   If it doesn't fit, create new bin.
        #   This maximizes the chance that the "last bin" (the one with the largest index) 
        #   is the one we've been filling, and it will be as full as possible.
        
        is_last_box = (idx == order[-1])
        
        if not bins:
            bins.append([[(0, 0, 0, w, h, d)], vol])
            continue
            
        last_bin_idx = len(bins) - 1
        last_pl, last_vol = bins[last_bin_idx]
        
        should_try_last_bin = False
        if is_last_box:
            should_try_last_bin = True
        else:
            if last_vol + vol <= target_vol:
                should_try_last_bin = True
            else:
                # Target reached, don't try last bin, create new one
                should_try_last_bin = False
        
        if should_try_last_bin:
            # Try to place in last bin
            anchors = _get_anchors(last_pl, W, H, D)
            # Sort anchors to prefer corners/faces that might fit better? 
            # Just iterate.
            found = False
            for cx, cy, cz in anchors:
                if _fits(last_pl, cx, cy, cz, w, h, d, W, H, D):
                    last_pl.append((cx, cy, cz, w, h, d))
                    last_vol += vol
                    bins[last_bin_idx] = (last_pl, last_vol)
                    found = True
                    break
            
            if found:
                continue # Placed, move to next box
            
            # If it didn't fit in last bin (geometrically), we must create a new bin
            # Even if volume was below target, geometry might prevent placement.
            # Or if is_last_box and it didn't fit, we create new bin.
            
        # If we get here, we are creating a new bin
        bins.append([[(0, 0, 0, w, h, d)], vol])
        
    return bins


def _score(bins, W, H, D):
    B = len(bins)
    if B == 0:
        return 0.0
    maxbi = max(i for i, (pl, bv) in enumerate(bins) if pl)
    last = bins[maxbi]
    C = W * H * D
    return B + 0.1 * (1.0 - (last[1] / C if C > 0 else 0.0))


def _orderings(boxes, n, rng):
    idx = list(range(n))
    # Volume descending is usually good for filling
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    # Max dimension descending
    yield sorted(idx, key=lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True)
    # Aspect ratio or other heuristics
    yield sorted(idx, key=lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], reverse=True)
    # Random shuffles
    for _ in range(10):
        s = idx[:]
        rng.shuffle(s)
        yield s


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    best_bins = None
    best_score = float('inf')
    t_start = time.time()
    rng = random.Random(42)
    
    # Try different target fill ratios and orderings
    # A lower target ratio means we open new bins sooner, potentially leading to a more packed last bin
    # if the last box happens to fit well into the previous "almost full" bin.
    # However, if we open a new bin for the last box, that bin is sparse.
    # The key is that the "last bin" is the one with the highest index.
    # If we pack bin 0 to 90%, bin 1 to 90%, and bin 2 gets the last box, bin 2 is sparse.
    # To fix this, we want the LAST box to go into a bin that is already somewhat full, 
    # OR we want the last bin to be the one that was being filled.
    
    # My logic in _pack_with_target:
    # If last bin is below target, put box there.
    # If last bin is above target, open new bin.
    # For the LAST box, always try to put it in the last bin (even if above target).
    # This ensures the last box joins the "current" bin, making it the last bin in the output.
    # Since the current bin was being filled towards the target, it should be reasonably full.
    
    # Let's iterate over target ratios
    target_ratios = [0.85, 0.90, 0.95, 0.98]
    
    for tr in target_ratios:
        if time.time() > DEADLINE:
            break
        for order in _orderings(boxes, n, rng):
            if time.time() > DEADLINE:
                break
            bins = _pack_with_target(boxes, container, order, target_fill_ratio=tr)
            s = _score(bins, W, H, D)
            if s < best_score:
                best_score = s
                best_bins = bins

    if best_bins is None:
        # Fallback: simple best fit
        best_bins = _pack_with_target(boxes, container, list(range(n)), target_fill_ratio=1.0)

    # Build placements
    assign = {}
    for bi, (pl, bv) in enumerate(best_bins):
        for (cx, cy, cz, w, h, d) in pl:
            key = (w, h, d)
            if key not in assign:
                assign[key] = []
            assign[key].append((bi, cx, cy, cz))
            
    placements = []
    used = {}
    for i, (w, h, d) in enumerate(boxes):
        key = (w, h, d)
        k = used.get(key, 0)
        if k < len(assign[key]):
            bi, cx, cy, cz = assign[key][k]
            used[key] = k + 1
            placements.append((bi, cx, cy, cz, w, h, d))
        else:
            # Fallback
            placements.append((999 + i, 0, 0, 0, w, h, d))
            
    return placements
