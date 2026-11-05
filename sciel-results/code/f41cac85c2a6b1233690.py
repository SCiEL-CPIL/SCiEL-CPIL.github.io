"""Search-based 3D bin packing with multiple strategies and consolidation."""
import time
import random
import math

DEADLINE_MARGIN = 10.0  # seconds before the 150s limit


def _overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    return (x1 < x2 + w2 and x2 < x1 + w1 and
            y1 < y2 + h2 and y2 < y1 + h1 and
            z1 < z2 + d2 and z2 < z1 + d1)


def _fits_in_bin(placed, x, y, z, w, h, d, W, H, D):
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
            return False
    return True


def _corner_positions(placed, W, H, D):
    """Generate candidate corner positions based on existing placements."""
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        # Right face
        candidates.append((px + pw, py, pz))
        # Front face
        candidates.append((px, py + ph, pz))
        # Top face
        candidates.append((px, py, pz + pd))
        # Edges
        candidates.append((px + pw, py + ph, pz))
        candidates.append((px + pw, py, pz + pd))
        candidates.append((px, py + ph, pz + pd))
        # Corner
        candidates.append((px + pw, py + ph, pz + pd))
    # Filter to valid positions within container
    valid = []
    seen = set()
    for (cx, cy, cz) in candidates:
        if cx >= 0 and cy >= 0 and cz >= 0 and cx < W and cy < H and cz < D:
            key = (cx, cy, cz)
            if key not in seen:
                seen.add(key)
                valid.append(key)
    return valid


def _best_fit_place(boxes, container, order, rng, max_probes=50):
    """Place boxes using best-fit strategy with corner candidates."""
    W, H, D = container
    n = len(boxes)
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    for idx in order:
        w, h, d = boxes[idx]
        vol = w * h * d
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        # Try to place in existing bins
        for bi, placed in enumerate(bins):
            candidates = _corner_positions(placed, W, H, D)
            # Add some random positions for diversity
            for _ in range(3):
                cx = rng.randint(0, max(0, W - w))
                cy = rng.randint(0, max(0, H - h))
                cz = rng.randint(0, max(0, D - d))
                candidates.append((cx, cy, cz))
            
            for (cx, cy, cz) in candidates:
                if _fits_in_bin(placed, cx, cy, cz, w, h, d, W, H, D):
                    # Score: prefer bins with more volume already, and positions closer to origin
                    bin_vol = sum(pw * ph * pd for (px, py, pz, pw, ph, pd) in placed)
                    pos_score = cx + cy + cz
                    score = -bin_vol + 0.01 * pos_score
                    if score < best_score:
                        best_score = score
                        best_bin = bi
                        best_pos = (cx, cy, cz)
        
        if best_bin >= 0:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        else:
            # Start a new bin
            bins.append([(0, 0, 0, w, h, d)])
    
    # Convert to placements format
    placements = []
    for bi, placed in enumerate(bins):
        for (x, y, z, w, h, d) in placed:
            placements.append((bi, x, y, z, w, h, d))
    return placements


def _shelf_pack(boxes, container, order):
    """Shelf-based packing: rows along x, shelves along y, layers along z."""
    W, H, D = container
    placements = []
    bins = []  # Each bin is a dict with x, y, z, row_h, layer_d
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        for bi in range(len(bins)):
            b = bins[bi]
            # Check if fits in current row
            if b['x'] + w <= W:
                # Check if fits in current shelf
                if b['y'] + h <= H:
                    # Check if fits in current layer
                    if b['z'] + d <= D:
                        placements.append((bi, b['x'], b['y'], b['z'], w, h, d))
                        b['x'] += w
                        b['row_h'] = max(b['row_h'], h)
                        b['layer_d'] = max(b['layer_d'], d)
                        placed = True
                        break
                else:
                    # Move to next shelf
                    b['y'] += b['row_h']
                    b['x'] = 0
                    b['row_h'] = 0
                    if b['y'] + h <= H:
                        if b['z'] + d <= D:
                            placements.append((bi, b['x'], b['y'], b['z'], w, h, d))
                            b['x'] += w
                            b['row_h'] = max(b['row_h'], h)
                            b['layer_d'] = max(b['layer_d'], d)
                            placed = True
                            break
                        else:
                            # Move to next layer
                            b['z'] += b['layer_d']
                            b['y'] = 0
                            b['x'] = 0
                            b['row_h'] = 0
                            b['layer_d'] = 0
                            if b['z'] + d <= D:
                                placements.append((bi, b['x'], b['y'], b['z'], w, h, d))
                                b['x'] += w
                                b['row_h'] = max(b['row_h'], h)
                                b['layer_d'] = max(b['layer_d'], d)
                                placed = True
                                break
                    else:
                        # Move to next layer
                        b['z'] += b['layer_d']
                        b['y'] = 0
                        b['x'] = 0
                        b['row_h'] = 0
                        b['layer_d'] = 0
                        if b['z'] + d <= D:
                            placements.append((bi, b['x'], b['y'], b['z'], w, h, d))
                            b['x'] += w
                            b['row_h'] = max(b['row_h'], h)
                            b['layer_d'] = max(b['layer_d'], d)
                            placed = True
                            break
            else:
                # Move to next row
                b['y'] += b['row_h']
                b['x'] = 0
                b['row_h'] = 0
                if b['y'] + h <= H:
                    if b['z'] + d <= D:
                        placements.append((bi, b['x'], b['y'], b['z'], w, h, d))
                        b['x'] += w
                        b['row_h'] = max(b['row_h'], h)
                        b['layer_d'] = max(b['layer_d'], d)
                        placed = True
                        break
                else:
                    # Move to next layer
                    b['z'] += b['layer_d']
                    b['y'] = 0
                    b['x'] = 0
                    b['row_h'] = 0
                    b['layer_d'] = 0
                    if b['z'] + d <= D:
                        placements.append((bi, b['x'], b['y'], b['z'], w, h, d))
                        b['x'] += w
                        b['row_h'] = max(b['row_h'], h)
                        b['layer_d'] = max(b['layer_d'], d)
                        placed = True
                        break
        
        if not placed:
            # Start a new bin
            bins.append({'x': 0, 'y': 0, 'z': 0, 'row_h': 0, 'layer_d': 0})
            placements.append((len(bins) - 1, 0, 0, 0, w, h, d))
            bins[-1]['x'] = w
            bins[-1]['row_h'] = h
            bins[-1]['layer_d'] = d
    
    return placements


def _score(placements, boxes, container):
    """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    if not placements:
        return float('inf')
    
    W, H, D = container
    container_vol = W * H * D
    
    # Find max bin label
    max_bin = max(p[0] for p in placements)
    
    # Count occupied bins
    occupied_bins = set(p[0] for p in placements)
    B = len(occupied_bins)
    
    # Compute last bin volume fraction
    last_bin_vol = 0
    for (bi, x, y, z, w, h, d) in placements:
        if bi == max_bin:
            last_bin_vol += w * h * d
    
    last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
    
    score = B + 0.1 * (1 - last_bin_frac)
    return score


def _consolidate(placements, boxes, container, max_moves=20):
    """Try to move boxes from earlier bins to later bins to reduce B."""
    W, H, D = container
    n = len(placements)
    
    # Group placements by bin
    bins = {}
    for i, (bi, x, y, z, w, h, d) in enumerate(placements):
        if bi not in bins:
            bins[bi] = []
        bins[bi].append(i)
    
    sorted_bins = sorted(bins.keys(), reverse=True)
    
    # Try to move boxes from lower bins to higher bins
    for src_bin in sorted(bins.keys()):
        if src_bin == sorted_bins[-1]:
            continue
        for idx in list(bins[src_bin]):
            bi, x, y, z, w, h, d = placements[idx]
            # Try to place in a higher bin
            for dst_bin in sorted_bins:
                if dst_bin <= src_bin:
                    continue
                # Check if fits in dst_bin
                dst_placed = [(placements[j][1], placements[j][2], placements[j][3],
                               placements[j][4], placements[j][5], placements[j][6])
                              for j in bins[dst_bin]]
                if _fits_in_bin(dst_placed, x, y, z, w, h, d, W, H, D):
                    # Move the box
                    placements[idx] = (dst_bin, x, y, z, w, h, d)
                    bins[src_bin].remove(idx)
                    if not bins[src_bin]:
                        del bins[src_bin]
                    bins[dst_bin].append(idx)
                    break
    
    return placements


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    
    # Try multiple strategies and keep the best
    best_placements = None
    best_score = float('inf')
    
    # Strategy 1: Best-fit with various orderings
    orderings = [
        list(range(n)),  # Original order
        sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True),  # Volume desc
        sorted(range(n), key=lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], reverse=True),  # Sum desc
        sorted(range(n), key=lambda i: max(boxes[i]), reverse=True),  # Max dim desc
        sorted(range(n), key=lambda i: min(boxes[i]), reverse=True),  # Min dim desc
        sorted(range(n), key=lambda i: boxes[i][0] / boxes[i][1] if boxes[i][1] > 0 else 0, reverse=True),  # Aspect ratio
    ]
    
    rng = random.Random(42)
    
    for order in orderings:
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        placements = _best_fit_place(boxes, container, order, rng, max_probes=30)
        score = _score(placements, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # Strategy 2: Shelf packing with various orderings
    for order in orderings:
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        placements = _shelf_pack(boxes, container, order)
        score = _score(placements, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # Consolidation: try to reduce B by moving boxes
    if best_placements is not None and time.time() - start_time < 140.0:
        consolidated = _consolidate(list(best_placements), boxes, container, max_moves=50)
        score = _score(consolidated, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = consolidated
    
    if best_placements is None:
        # Fallback: simple shelf
        best_placements = _shelf_pack(boxes, container, list(range(n)))
    
    return best_placements
