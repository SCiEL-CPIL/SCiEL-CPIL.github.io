import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    if container_vol <= 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort by volume descending, then by max dimension descending, then by min dimension
    indexed = list(range(n))
    def sort_key(i):
        w, h, d = boxes[i]
        vol = w * h * d
        mx = max(w, h, d)
        mn = min(w, h, d)
        return (-vol, -mx, -mn, i)
    indexed.sort(key=sort_key)
    
    # bins: list of lists of (x, y, z, w, h, d)
    bins = []
    results = [None] * n
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True
    
    def generate_candidates(bin_boxes, w, h, d):
        """Generate candidate positions for placing a box in a bin."""
        candidates = set()
        candidates.add((0, 0, 0))
        
        if not bin_boxes:
            return [(0, 0, 0)]
        
        # Collect unique x, y, z coordinates from existing boxes
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            xs.add(bx)
            xs.add(bx + bw)
            ys.add(by)
            ys.add(by + bh)
            zs.add(bz)
            zs.add(bz + bd)
        
        # Filter to valid range
        xs = sorted(set(x for x in xs if 0 <= x <= W - w))
        ys = sorted(set(y for y in ys if 0 <= y <= H - h))
        zs = sorted(set(z for z in zs if 0 <= z <= D - d))
        
        # If too many combinations, use a smarter subset
        if len(xs) * len(ys) * len(zs) > 3000:
            # Use only the most promising: corners of each box
            for bx, by, bz, bw, bh, bd in bin_boxes:
                for cx in [bx, bx + bw]:
                    for cy in [by, by + bh]:
                        for cz in [bz, bz + bd]:
                            if 0 <= cx <= W - w and 0 <= cy <= H - h and 0 <= cz <= D - d:
                                candidates.add((cx, cy, cz))
            # Also add the axis-aligned neighbors
            for bx, by, bz, bw, bh, bd in bin_boxes:
                if bx + bw <= W - w:
                    candidates.add((bx + bw, by, bz))
                if by + bh <= H - h:
                    candidates.add((bx, by + bh, bz))
                if bz + bd <= D - d:
                    candidates.add((bx, by, bz + bd))
        else:
            for cx in xs:
                for cy in ys:
                    for cz in zs:
                        candidates.add((cx, cy, cz))
        
        # Sort by compactness: prefer lower x, then y, then z
        return sorted(candidates, key=lambda p: (p[0], p[1], p[2]))
    
    def find_best_position(bin_boxes, w, h, d):
        """Find the best position in a bin, minimizing waste (bounding box growth)."""
        if not bin_boxes:
            if can_place(bin_boxes, 0, 0, 0, w, h, d):
                return (0, 0, 0)
            return None
        
        # Compute current bounding box of the bin
        cur_max_x = max(bx + bw for bx, by, bz, bw, bh, bd in bin_boxes)
        cur_max_y = max(by + bh for bx, by, bz, bw, bh, bd in bin_boxes)
        cur_max_z = max(bz + bd for bx, by, bz, bw, bh, bd in bin_boxes)
        cur_vol = cur_max_x * cur_max_y * cur_max_z
        
        candidates = generate_candidates(bin_boxes, w, h, d)
        
        best_pos = None
        best_score = float('inf')
        
        for x, y, z in candidates:
            if not can_place(bin_boxes, x, y, z, w, h, d):
                continue
            new_max_x = max(cur_max_x, x + w)
            new_max_y = max(cur_max_y, y + h)
            new_max_z = max(cur_max_z, z + d)
            new_vol = new_max_x * new_max_y * new_max_z
            # Score: new bounding volume (lower is better)
            if new_vol < best_score:
                best_score = new_vol
                best_pos = (x, y, z)
        
        return best_pos
    
    def bin_fill_ratio(bin_boxes):
        """Compute the fill ratio of a bin (volume used / bounding box volume)."""
        if not bin_boxes:
            return 0.0
        cur_max_x = max(bx + bw for bx, by, bz, bw, bh, bd in bin_boxes)
        cur_max_y = max(by + bh for bx, by, bz, bw, bh, bd in bin_boxes)
        cur_max_z = max(bz + bd for bx, by, bz, bw, bh, bd in bin_boxes)
        bb_vol = cur_max_x * cur_max_y * cur_max_z
        if bb_vol == 0:
            return 0.0
        used_vol = sum(bw * bh * bd for _, _, _, bw, bh, bd in bin_boxes)
        return used_vol / bb_vol
    
    # Main packing loop
    for idx in indexed:
        if time.time() > deadline:
            # Emergency: place remaining boxes one per bin starting from current bin count
            for remaining_idx in indexed[indexed.index(idx):]:
                b = len(bins)
                bins.append([(0, 0, 0, boxes[remaining_idx][0], boxes[remaining_idx][1], boxes[remaining_idx][2])])
                results[remaining_idx] = (b, 0, 0, 0, boxes[remaining_idx][0], boxes[remaining_idx][1], boxes[remaining_idx][2])
            break
        
        w, h, d = boxes[idx]
        placed = False
        
        # Try to fit into existing bins, preferring bins with higher fill ratio (best-fit decreasing)
        if bins:
            bin_order = sorted(range(len(bins)), key=lambda bi: -bin_fill_ratio(bins[bi]))
            for b_idx in bin_order:
                pos = find_best_position(bins[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    results[idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Start a new bin
            b_idx = len(bins)
            x, y, z = 0, 0, 0
            bins.append([(x, y, z, w, h, d)])
            results[idx] = (b_idx, x, y, z, w, h, d)
    
    return results
