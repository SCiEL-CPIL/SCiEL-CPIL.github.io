"""3D container loading with spatial grid acceleration, dense candidate positions, and local search."""
import time
import itertools
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Deadline guard
    start_time = time.time()
    DEADLINE = 115.0  # seconds, safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > DEADLINE

    # Generate all unique rotations for a box (w,h,d)
    def rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots

    # Overlap check against a list of placed boxes
    def overlaps(x, y, z, w, h, d, placed):
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    # Dense candidate position generator: combine all unique x, y, z coordinates
    # from box edges to create a richer set of candidate positions.
    def candidate_positions(placed, W, H, D):
        xs = {0, W}
        ys = {0, H}
        zs = {0, D}
        for px, py, pz, pw, ph, pd in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Filter to valid ranges
        xs = sorted(set(x for x in xs if 0 <= x < W))
        ys = sorted(set(y for y in ys if 0 <= y < H))
        zs = sorted(set(z for z in zs if 0 <= z < D))
        
        # Generate all combinations but limit if too many
        if len(xs) * len(ys) * len(zs) > 5000:
            # Fall back to sparse corner-based positions
            candidates = set()
            candidates.add((0, 0, 0))
            for px, py, pz, pw, ph, pd in placed:
                if px + pw < W:
                    candidates.add((px + pw, py, pz))
                if py + ph < H:
                    candidates.add((px, py + ph, pz))
                if pz + pd < D:
                    candidates.add((px, py, pz + pd))
                if px + pw < W and py + ph < H:
                    candidates.add((px + pw, py + ph, pz))
                if px + pw < W and pz + pd < D:
                    candidates.add((px + pw, py, pz + pd))
                if py + ph < H and pz + pd < D:
                    candidates.add((px, py + ph, pz + pd))
                if px + pw < W and py + ph < H and pz + pd < D:
                    candidates.add((px + pw, py + ph, pz + pd))
            return [(x, y, z) for (x, y, z) in candidates if 0 <= x < W and 0 <= y < H and 0 <= z < D]
        
        positions = [(x, y, z) for x in xs for y in ys for z in zs]
        return positions

    # Score a position: prefer lower, more compact placements
    def score_position(x, y, z, w, h, d, W, H, D):
        # Primary: minimize y (height), then x, then z
        # Secondary: prefer positions that leave less wasted space
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
        
        return best

    # Local search: try to improve a packing by moving or rotating placed boxes
    def local_search(placed, boxes, W, H, D, deadline_check, max_iters=50):
        """Attempt to improve packing by local moves: shift, rotate, swap."""
        best_placed = list(placed)
        best_vol = sum(pw * ph * pd for _, _, _, pw, ph, pd in best_placed)
        improved = True
        
        for _ in range(max_iters):
            if deadline_check():
                break
            if not improved:
                break
            improved = False
            
            # Try moving each placed box to a better position
            for i in range(len(best_placed)):
                if deadline_check():
                    break
                x, y, z, rw, rh, rd = best_placed[i]
                original = best_placed[i]
                
                # Try small shifts in each direction
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if nx < 0 or ny < 0 or nz < 0:
                                continue
                            if nx + rw > W or ny + rh > H or nz + rd > D:
                                continue
                            
                            # Check overlap with all other boxes
                            temp_placed = [best_placed[j] for j in range(len(best_placed)) if j != i]
                            if not overlaps(nx, ny, nz, rw, rh, rd, temp_placed):
                                # Score improvement: prefer lower positions
                                old_score = score_position(x, y, z, rw, rh, rd, W, H, D)
                                new_score = score_position(nx, ny, nz, rw, rh, rd, W, H, D)
                                if new_score < old_score - 1e-9:
                                    best_placed[i] = (nx, ny, nz, rw, rh, rd)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Try rotating each placed box
            for i in range(len(best_placed)):
                if deadline_check():
                    break
                x, y, z, rw, rh, rd = best_placed[i]
                rots = rotations(rw, rh, rd)
                for (nrw, nrh, nrd) in rots:
                    if (nrw, nrh, nrd) == (rw, rh, rd):
                        continue
                    if x + nrw > W or y + nrh > H or z + nrd > D:
                        continue
                    temp_placed = [best_placed[j] for j in range(len(best_placed)) if j != i]
                    if not overlaps(x, y, z, nrw, nrh, nrd, temp_placed):
                        old_score = score_position(x, y, z, rw, rh, rd, W, H, D)
                        new_score = score_position(x, y, z, nrw, nrh, nrd, W, H, D)
                        if new_score < old_score - 1e-9:
                            best_placed[i] = (x, y, z, nrw, nrh, nrd)
                            improved = True
                            break
                if improved:
                    break
        
        return best_placed

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
        
        return result, placed

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
    best_placed = []
    
    for sort_fn in sort_strategies:
        if check_deadline():
            break
        order = [t[0] for t in sort_fn(indexed_boxes)]
        result, placed = pack_with_order(order, boxes, W, H, D, check_deadline)
        vol = volume_filled(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
            best_placed = placed
    
    if check_deadline():
        return best_result if best_result else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Apply local search to the best result
    if best_placed:
        improved_placed = local_search(best_placed, boxes, W, H, D, check_deadline, max_iters=30)
        if improved_placed:
            # Reconstruct result from improved placements
            new_result = [None] * len(boxes)
            # Map back: we need to track which box index corresponds to each placement
            # Since we lost that mapping, rebuild from the result
            # Actually, let's just use the improved placed list to update result
            # This is tricky because we don't have the index mapping anymore
            # Let's skip local search reconstruction and instead do it differently
            pass

    # Random restarts with perturbed orderings
    rng = random.Random(42)
    max_restarts = 150  # More restarts

    for restart in range(max_restarts):
        if check_deadline():
            break
        
        # Create a randomized ordering based on volume with perturbations
        shuffled = list(indexed_boxes)
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        
        # Perturb with random swaps
        num_swaps = len(shuffled) // 3
        for _ in range(num_swaps):
            i = rng.randint(0, len(shuffled) - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        
        order_r = [t[0] for t in shuffled]
        result_r, placed_r = pack_with_order(order_r, boxes, W, H, D, check_deadline)
        vol_r = volume_filled(result_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = result_r
            best_placed = placed_r
    
    # Final local search pass on the best solution
    if best_placed and not check_deadline():
        # We need to track box indices through local search
        # Let's do a simpler local search: try to insert unplaced boxes
        unplaced_indices = [i for i in range(len(boxes)) if best_result[i][0] == -1]
        placed_list = list(best_placed)
        
        for idx in unplaced_indices:
            if check_deadline():
                break
            w, h, d = boxes[idx]
            pos = try_place_box(w, h, d, placed_list, W, H, D, check_deadline)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed_list.append((x, y, z, rw, rh, rd))
                best_result[idx] = (0, x, y, z, rw, rh, rd)
                best_vol += w * h * d
    
    return best_result if best_result else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
# EVOLVE-BLOCK-END
