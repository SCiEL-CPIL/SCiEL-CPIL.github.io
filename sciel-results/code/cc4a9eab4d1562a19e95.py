import time
import random

def pack(boxes, container):
    """
    Place every box into numbered bins.
    Returns list of (bin, x, y, z, w, h, d) for each box in original order.
    Score: mean B + 0.1*(1 - last-bin occupied volume fraction), lower is better.
    """
    DEADLINE = 115.0  # Safety margin before 150s kill (allowing for overhead)
    START = time.time()
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Pre-sort strategies
    strategies = []
    
    # 1. Sort by volume descending
    vol_sorted = sorted(range(n), key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    strategies.append(vol_sorted)
    
    # 2. Sort by max dimension descending
    max_dim_sorted = sorted(range(n), key=lambda i: -max(boxes[i]))
    strategies.append(max_dim_sorted)
    
    # 3. Sort by (w) descending, then (h), then (d)
    whd_sorted = sorted(range(n), key=lambda i: (-boxes[i][0], -boxes[i][1], -boxes[i][2]))
    strategies.append(whd_sorted)
    
    # 4. Sort by (h) descending, then (w), then (d) - for shelf packing
    hw_sorted = sorted(range(n), key=lambda i: (-boxes[i][1], -boxes[i][0], -boxes[i][2]))
    strategies.append(hw_sorted)
    
    # 5. Sort by (d) descending
    d_sorted = sorted(range(n), key=lambda i: (-boxes[i][2], -boxes[i][1], -boxes[i][0]))
    strategies.append(d_sorted)
    
    # 6. Reverse of volume (small first)
    vol_asc = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]))
    strategies.append(vol_asc)
    
    best_placements = None
    best_score = float('inf')
    
    def evaluate(placements):
        """Compute score: mean B + 0.1*(1 - last-bin occupied volume / container volume)"""
        if not placements:
            return 0.0
        bins = {}
        for b, x, y, z, w, h, d in placements:
            if b not in bins:
                bins[b] = 0
            bins[b] += w * h * d
        
        if not bins:
            return 0.0
        
        B = len(bins)
        last_bin = max(bins.keys())
        last_vol = bins[last_bin]
        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        return B + 0.1 * (1.0 - last_vol / container_vol)
    
    def shelf_pack(order):
        """Shelf-based packing with given box order."""
        placements = [None] * n
        b = 0
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            
            # Try to place in current position
            placed = False
            
            # Check if fits in current shelf/layer/bin
            if x + w <= W and y + h <= H and z + d <= D:
                placements[idx] = (b, x, y, z, w, h, d)
                x += w
                row_h = max(row_h, h)
                layer_d = max(layer_d, d)
                placed = True
            else:
                # Try next shelf in current layer
                if y + row_h + h <= H and z + d <= D:
                    x, y, row_h = 0, y + row_h, h
                    placements[idx] = (b, x, y, z, w, h, d)
                    x = w
                    layer_d = max(layer_d, d)
                    placed = True
                else:
                    # Try next layer
                    if z + layer_d + d <= D:
                        x, y, z, row_h, layer_d = 0, 0, z + layer_d, h, d
                        placements[idx] = (b, x, y, z, w, h, d)
                        x = w
                        placed = True
                    else:
                        # New bin
                        b += 1
                        x, y, z, row_h, layer_d = 0, 0, 0, h, d
                        placements[idx] = (b, x, y, z, w, h, d)
                        x = w
                        placed = True
        
        return placements
    
    def best_fit_pack(order):
        """Best-fit decreasing style: for each box, find the bin+position that minimizes waste."""
        placements = [None] * n
        # For each bin, maintain a list of placed boxes and their bounding box
        bin_occupied = {}  # bin_id -> list of (x, y, z, w, h, d)
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_metric = float('inf')
            
            # Check existing bins
            for b_id in sorted(bin_occupied.keys()):
                # Find best corner in this bin
                # Use corner-based placement: try all corners formed by existing boxes
                corners = [(0, 0, 0)]
                for (px, py, pz, pw, ph, pd) in bin_occupied[b_id]:
                    corners.append((px + pw, py, pz))
                    corners.append((px, py + ph, pz))
                    corners.append((px, py, pz + pd))
                    corners.append((px + pw, py + ph, pz))
                    corners.append((px + pw, py, pz + pd))
                    corners.append((px, py + ph, pz + pd))
                    corners.append((px + pw, py + ph, pz + pd))
                
                for (cx, cy, cz) in corners:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    # Check overlap
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in bin_occupied[b_id]:
                        if not (cx + w <= px or px + pw <= cx or
                                cy + h <= py or py + ph <= cy or
                                cz + d <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    if not overlap:
                        # Metric: prefer lower z, then lower y, then lower x (fill bottom first)
                        metric = cz * 100000 + cy * 1000 + cx
                        if metric < best_metric:
                            best_metric = metric
                            best_pos = (b_id, cx, cy, cz)
            
            # If no valid position in existing bins, create new bin
            if best_pos is None:
                new_b = max(bin_occupied.keys()) + 1 if bin_occupied else 0
                bin_occupied[new_b] = []
                best_pos = (new_b, 0, 0, 0)
            
            b_id, cx, cy, cz = best_pos
            placements[idx] = (b_id, cx, cy, cz, w, h, d)
            bin_occupied[b_id].append((cx, cy, cz, w, h, d))
        
        return placements
    
    # Run each strategy with both shelf and best-fit
    for order in strategies:
        if time.time() > DEADLINE:
            break
        
        # Shelf packing
        placements = shelf_pack(order)
        if all(p is not None for p in placements):
            score = evaluate(placements)
            if score < best_score:
                best_score = score
                best_placements = placements
        
        # Best-fit packing (slower but potentially better)
        if time.time() > DEADLINE - 20:
            break
        placements = best_fit_pack(order)
        if all(p is not None for p in placements):
            score = evaluate(placements)
            if score < best_score:
                best_score = score
                best_placements = placements
    
    # If we still have time, try a local improvement: re-optimize bin assignment
    if best_placements is None:
        # Fallback: simple shelf with volume sort
        placements = shelf_pack(vol_sorted)
        best_placements = placements
        best_score = evaluate(placements)
    
    # Verify no overlaps and all within container
    def verify(placements):
        if not placements:
            return True
        bin_boxes = {}
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            if b not in bin_boxes:
                bin_boxes[b] = []
            bin_boxes[b].append((x, y, z, w, h, d))
            if x < 0 or y < 0 or z < 0 or x + w > W or y + h > H or z + d > D:
                return False
        for b, blist in bin_boxes.items():
            for i in range(len(blist)):
                for j in range(i+1, len(blist)):
                    x1, y1, z1, w1, h1, d1 = blist[i]
                    x2, y2, z2, w2, h2, d2 = blist[j]
                    if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                            y1 + h1 <= y2 or y2 + h2 <= y1 or
                            z1 + d1 <= z2 or z2 + d2 <= z1):
                        return False
        return True
    
    if not verify(best_placements):
        # Fallback to a simple valid placement if something went wrong
        # Just put each box in its own bin at (0,0,0)
        best_placements = [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_placements
