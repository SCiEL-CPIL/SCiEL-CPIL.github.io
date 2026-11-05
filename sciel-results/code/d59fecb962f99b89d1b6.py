import time
import itertools
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute box properties
    box_vols = [w * h * d for w, h, d in boxes]
    
    # Generate multiple sort orders to explore different packing strategies
    orders = []
    # Order 1: volume descending
    orders.append(sorted(range(n), key=lambda i: box_vols[i], reverse=True))
    # Order 2: max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    # Order 3: sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    # Order 4: most cube-like first (lowest aspect ratio)
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]) / min(boxes[i])))
    # Order 5: height (z-dim) ascending - for layer packing
    orders.append(sorted(range(n), key=lambda i: min(boxes[i])))
    
    best_result = None
    best_filled = 0.0
    
    def get_orientations(w, h, d):
        """Return unique orientations sorted by height (z) ascending."""
        orients = list(set(itertools.permutations([w, h, d])))
        orients.sort(key=lambda o: (o[2], o[1], o[0]))
        return orients
    
    def fits(x, y, z, w, h, d, placed):
        """Check if box fits in container and doesn't overlap with placed boxes."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            # No overlap if separated in any dimension
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def generate_candidates(placed, W, H, D):
        """Generate candidate positions from placed boxes, sorted by (z, y, x)."""
        xs = {0}
        ys = {0}
        zs = {0}
        for px, py, pz, pw, ph, pd in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Filter to valid range
        xs = sorted(x for x in xs if 0 <= x <= W)
        ys = sorted(y for y in ys if 0 <= y <= H)
        zs = sorted(z for z in zs if 0 <= z <= D)
        
        # Generate combinations, sort by (z, y, x) to prefer bottom-front-left
        cands = []
        for z in zs:
            for y in ys:
                for x in xs:
                    cands.append((x, y, z))
        return cands
    
    def try_pack(order, deadline):
        """Try to pack boxes in given order; return placements and filled volume."""
        placed = []
        result = [None] * n
        filled_vol = 0.0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            best_placement = None
            best_cost = float('inf')
            
            orientations = get_orientations(w, h, d)
            
            # Generate candidate positions
            candidates = generate_candidates(placed, W, H, D)
            
            # Limit candidates to avoid excessive computation
            MAX_CANDS = min(len(candidates), 200)
            
            for cx, cy, cz in candidates[:MAX_CANDS]:
                for ow, oh, od in orientations:
                    if not fits(cx, cy, cz, ow, oh, od, placed):
                        continue
                    # Cost: prioritize low z, then low y, then low x
                    cost = cz * (H + 1) * (W + 1) + cy * (W + 1) + cx
                    if cost < best_cost:
                        best_cost = cost
                        best_placement = (cx, cy, cz, ow, oh, od)
            
            if best_placement:
                bx, by, bz, bw, bh, bd = best_placement
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled_vol += box_vols[idx]
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill remaining with -1
        for i in range(n):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result, filled_vol
    
    # Try each order and keep the best
    for order in orders:
        if time.time() > deadline:
            break
        
        result, filled_vol = try_pack(order, deadline)
        
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = result
        
        # If we've used most of our time, stop
        if time.time() > deadline - 5:
            break
    
    # If no result found, return all as left out
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
