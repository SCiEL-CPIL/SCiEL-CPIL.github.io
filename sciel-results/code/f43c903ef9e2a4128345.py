"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Deadline guard: keep a safety margin before the 120s/150s limit.
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > DEADLINE

    # Generate all 6 unique rotations for a box (w,h,d)
    def rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots

    # Check if a candidate box (w,h,d) at (x,y,z) overlaps any placed box
    def overlaps(x, y, z, w, h, d, placed):
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    # Generate candidate positions using "bottom-left-far" style heuristics
    # We use a set of corner-based candidate positions to reduce search space.
    def candidate_positions(placed, W, H, D):
        """Generate candidate (x,y,z) positions based on existing placements."""
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))
        # For each placed box, generate positions adjacent to it
        for px, py, pz, pw, ph, pd in placed:
            # Right of box
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            # Above box
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            # In front of box
            if pz + pd < D:
                candidates.add((px, py, pz + pd))
            # Diagonal combinations (6-face corners)
            if px + pw < W and py + ph < H:
                candidates.add((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D:
                candidates.add((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D:
                candidates.add((px, py + ph, pz + pd))
            if px + pw < W and py + ph < H and pz + pd < D:
                candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter out invalid positions
        valid = []
        for (x, y, z) in candidates:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        
        return valid

    # Score a placement: prefer positions that are lower, more to the left, and more to the front
    def score_position(x, y, z, w, h, d, W, H, D):
        # Weighted score: prefer low y, low x, low z
        # Also prefer positions that leave less "wasted" space above
        return (y * 1.0 + x * 0.5 + z * 0.3) + (h / H) * 0.1 + (w / W) * 0.05 + (d / D) * 0.02

    # Try to place a single box with all rotations, return best placement or None
    def try_place_box(w, h, d, placed, W, H, D, deadline_check):
        rots = rotations(w, h, d)
        best = None
        best_score = float('inf')
        
        # Generate candidate positions
        positions = candidate_positions(placed, W, H, D)
        
        for (rw, rh, rd) in rots:
            for (x, y, z) in positions:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                if overlaps(x, y, z, rw, rh, rd, placed):
                    continue
                sc = score_position(x, y, z, rw, rh, rd, W, H, D)
                if sc < best_score:
                    best_score = sc
                    best = (x, y, z, rw, rh, rd)
                if deadline_check():
                    break
            if deadline_check():
                break
        
        return best

    # Sort boxes by a heuristic: larger boxes first (by volume, then by max dimension)
    def sort_boxes(box_list):
        # box_list is list of (index, w, h, d)
        return sorted(box_list, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3]), -sum(t[1:4])/3))

    # Main packing strategy: try multiple orderings and keep the best
    def pack_with_order(order, boxes, W, H, D, deadline_check):
        """Try to pack boxes in the given order. Returns list of placements."""
        placed = []  # list of (x, y, z, w, h, d)
        result = [None] * len(boxes)  # result[i] = (bin, x, y, z, w, h, d)
        
        for idx in order:
            w, h, d = boxes[idx]
            if deadline_check():
                break
            pos = try_place_box(w, h, d, placed, W, H, D, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
        
        # Mark unplaced boxes
        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result

    def volume_filled(placements):
        total = 0
        for (b, x, y, z, w, h, d) in placements:
            if b == 0:
                total += w * h * d
        return total

    # Prepare indexed boxes
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Strategy 1: Sort by volume descending
    order1 = [t[0] for t in sort_boxes(indexed_boxes)]
    result1 = pack_with_order(order1, boxes, W, H, D, check_deadline)
    best_result = result1
    best_vol = volume_filled(result1)
    
    if check_deadline():
        return best_result
    
    # Strategy 2: Sort by max dimension descending
    order2 = sorted(indexed_boxes, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
    order2 = [t[0] for t in order2]
    if order2 != order1:
        result2 = pack_with_order(order2, boxes, W, H, D, check_deadline)
        vol2 = volume_filled(result2)
        if vol2 > best_vol:
            best_vol = vol2
            best_result = result2
    
    if check_deadline():
        return best_result
    
    # Strategy 3: Sort by surface area descending
    order3 = sorted(indexed_boxes, key=lambda t: (-2*(t[1]*t[2] + t[2]*t[3] + t[1]*t[3]), -t[1]*t[2]*t[3]))
    order3 = [t[0] for t in order3]
    if order3 != order1 and order3 != order2:
        result3 = pack_with_order(order3, boxes, W, H, D, check_deadline)
        vol3 = volume_filled(result3)
        if vol3 > best_vol:
            best_vol = vol3
            best_result = result3
    
    if check_deadline():
        return best_result
    
    # Strategy 4-6: Random restarts with shuffled orderings
    rng = random.Random(42)
    for restart in range(20):
        if check_deadline():
            break
        # Create a randomized ordering: mix of volume-sorted and random
        shuffled = list(indexed_boxes)
        rng.shuffle(shuffled)
        # Partially sort: sort top half by volume, keep bottom half random
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        # Perturb: swap some adjacent elements
        for _ in range(len(shuffled) // 4):
            i = rng.randint(0, len(shuffled) - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        order_r = [t[0] for t in shuffled]
        result_r = pack_with_order(order_r, boxes, W, H, D, check_deadline)
        vol_r = volume_filled(result_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r
    
    return best_result
# EVOLVE-BLOCK-END
