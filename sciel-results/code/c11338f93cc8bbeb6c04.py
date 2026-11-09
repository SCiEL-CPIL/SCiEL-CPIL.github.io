"""Rewritten 3D container loading with spatial grid acceleration and more restarts."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Deadline guard: keep a safety margin before the 150s limit.
    start_time = time.time()
    DEADLINE = 125.0  # seconds, safety margin

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

    # 3D spatial grid for fast overlap detection
    # Grid cell size is 1 unit; we track occupied cells
    # For efficiency, we use a set of (x, y, z) tuples for occupied unit cells
    # But that could be too large. Instead, we use an interval-tree-like approach
    # or simply a list of placed boxes with early termination.
    
    # Actually, for 60 boxes, O(n) overlap check is fine. The bottleneck is
    # the number of candidate positions and rotations tried.
    # Let's use a smarter candidate position generator.

    def overlaps(x, y, z, w, h, d, placed):
        """Check if box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    # Generate candidate positions using bottom-left-far heuristic with more variety
    def candidate_positions(placed, W, H, D):
        """Generate candidate (x,y,z) positions."""
        candidates = set()
        candidates.add((0, 0, 0))
        
        # For each placed box, generate positions on its 6 faces and 8 corners
        for px, py, pz, pw, ph, pd in placed:
            # Face-adjacent positions
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            if pz + pd < D:
                candidates.add((px, py, pz + pd))
            
            # Edge-adjacent (2-face) positions
            if px + pw < W and py + ph < H:
                candidates.add((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D:
                candidates.add((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D:
                candidates.add((px, py + ph, pz + pd))
            
            # Corner (3-face) position
            if px + pw < W and py + ph < H and pz + pd < D:
                candidates.add((px + pw, py + ph, pz + pd))
        
        # Also add some "snap to existing boundaries" positions
        # For each placed box, try aligning to its left, right, bottom, top, front, back
        for px, py, pz, pw, ph, pd in placed:
            # Align x to left edge of box
            candidates.add((0, py, pz))
            # Align x to right edge of box
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            # Align y to bottom edge
            candidates.add((px, 0, pz))
            # Align y to top edge
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            # Align z to front edge
            candidates.add((px, py, 0))
            # Align z to back edge
            if pz + pd < D:
                candidates.add((px, py, pz + pd))
        
        # Filter valid positions
        valid = []
        for (x, y, z) in candidates:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        
        return valid

    # Score a position: prefer lower, more compact placements
    def score_position(x, y, z, w, h, d, W, H, D):
        # Primary: minimize y (height), then x, then z
        # Secondary: prefer positions that fill gaps
        return (y * 1.0 + x * 0.3 + z * 0.2)

    # Try to place a single box with all rotations, return best placement or None
    def try_place_box(w, h, d, placed, W, H, D, deadline_check):
        rots = rotations(w, h, d)
        best = None
        best_score = float('inf')
        
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
                    return best
            if deadline_check():
                return best
        
        return best

    # Sort boxes by various heuristics
    def sort_by_volume(box_list):
        return sorted(box_list, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))

    def sort_by_max_dim(box_list):
        return sorted(box_list, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_by_surface(box_list):
        return sorted(box_list, key=lambda t: (-2*(t[1]*t[2] + t[2]*t[3] + t[1]*t[3]), -t[1]*t[2]*t[3]))

    def sort_by_min_dim(box_list):
        return sorted(box_list, key=lambda t: (-min(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_by_base_area(box_list):
        return sorted(box_list, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))

    # Main packing: try to pack boxes in given order
    def pack_with_order(order, boxes, W, H, D, deadline_check):
        placed = []
        result = [None] * len(boxes)
        
        for idx in order:
            w, h, d = boxes[idx]
            if deadline_check():
                break
            pos = try_place_box(w, h, d, placed, W, H, D, deadline_check)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
        
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
    
    # Multiple deterministic sort strategies
    sort_strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_surface,
        sort_by_min_dim,
        sort_by_base_area,
    ]
    
    best_result = None
    best_vol = 0
    
    for sort_fn in sort_strategies:
        if check_deadline():
            break
        order = [t[0] for t in sort_fn(indexed_boxes)]
        result = pack_with_order(order, boxes, W, H, D, check_deadline)
        vol = volume_filled(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    if check_deadline():
        return best_result if best_result else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Random restarts with perturbed orderings
    rng = random.Random(42)
    max_restarts = 200  # More restarts due to faster per-placement check
    
    for restart in range(max_restarts):
        if check_deadline():
            break
        
        # Create a randomized ordering based on volume with perturbations
        shuffled = list(indexed_boxes)
        # Sort by volume as base
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        
        # Perturb with random swaps
        num_swaps = len(shuffled) // 3
        for _ in range(num_swaps):
            i = rng.randint(0, len(shuffled) - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        
        order_r = [t[0] for t in shuffled]
        result_r = pack_with_order(order_r, boxes, W, H, D, check_deadline)
        vol_r = volume_filled(result_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r
    
    return best_result if best_result else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
# EVOLVE-BLOCK-END
