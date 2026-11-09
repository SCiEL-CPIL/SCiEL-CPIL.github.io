"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Generate all 6 orientations for a box (unique)
    def get_orientations(w, h, d):
        dims = list(itertools.permutations((w, h, d)))
        seen = set()
        unique = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                unique.append(dim)
        return unique
    
    # Pre-compute orientations
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(get_orientations(w, h, d))
    
    # Volume and other features
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # Sort indices by various criteria
    order_vol_desc = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    order_max_dim_desc = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    order_aspect_asc = sorted(range(n), key=lambda i: max(boxes[i]) / min(boxes[i]) if min(boxes[i]) > 0 else float('inf'))
    
    def can_place(placed_set, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) at position (x, y, z) overlaps with any placed box."""
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        if x < 0 or y < 0 or z < 0:
            return False
        for (px, py, pz, pw, ph, pd) in placed_set:
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True
    
    def find_position(placed_set, bw, bh, bd):
        """Find the best position for a box using corner-based heuristic."""
        candidates = [(0, 0, 0)]
        
        for (px, py, pz, pw, ph, pd) in placed_set:
            # All 8 corners of the box, offset to be adjacent
            candidates.extend([
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
                (px + pw, py + ph, pz),
                (px + pw, py, pz + pd),
                (px, py + ph, pz + pd),
                (px + pw, py + ph, pz + pd),
            ])
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in unique_candidates:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            if not can_place(placed_set, cx, cy, cz, bw, bh, bd):
                continue
            
            # Score: prefer lower z, then lower y, then lower x
            score = cz * 10000 + cy * 100 + cx
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    def run_pack(order, orientations_override=None):
        """Try to pack boxes in the given order."""
        placed = []
        results = [None] * n
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = orientations_override[idx] if orientations_override else box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    score = cz * 10000 + cy * 100 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results
    
    def compute_volume(results):
        return sum(w * h * d for (b, x, y, z, w, h, d) in results if b == 0)
    
    def try_reinsert(placed, results, left_out_indices):
        """Try to re-insert left-out boxes."""
        if not left_out_indices:
            return results, placed
        
        for li_idx in left_out_indices:
            bw, bh, bd = boxes[li_idx]
            orientations = box_orientations[li_idx]
            
            for (ow, oh, od) in orientations:
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    placed.append((cx, cy, cz, ow, oh, od))
                    results[li_idx] = (0, cx, cy, cz, ow, oh, od)
                    return results, placed
        
        return results, placed
    
    # Try multiple packing orders
    best_results = None
    best_volume = -1
    
    # Order 1: Volume descending
    results1 = run_pack(order_vol_desc)
    vol1 = compute_volume(results1)
    if vol1 > best_volume:
        best_volume = vol1
        best_results = results1
    
    if time.time() - start_time > DEADLINE - 10:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 2: Max dimension descending
    results2 = run_pack(order_max_dim_desc)
    vol2 = compute_volume(results2)
    if vol2 > best_volume:
        best_volume = vol2
        best_results = results2
    
    if time.time() - start_time > DEADLINE - 10:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 3: Aspect ratio ascending (cubic first)
    results3 = run_pack(order_aspect_asc)
    vol3 = compute_volume(results3)
    if vol3 > best_volume:
        best_volume = vol3
        best_results = results3
    
    if time.time() - start_time > DEADLINE - 10:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Try random orders
    random.seed(42)
    for trial in range(300):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(n))
        random.shuffle(order)
        results = run_pack(order)
        vol = compute_volume(results)
        if vol > best_volume:
            best_volume = vol
            best_results = results
    
    # Try perturbed volume-descending orders
    if best_results is not None and time.time() - start_time < DEADLINE - 10:
        for trial in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
            order = list(order_vol_desc)
            for _ in range(random.randint(1, 5)):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                order[i], order[j] = order[j], order[i]
            
            results = run_pack(order)
            vol = compute_volume(results)
            if vol > best_volume:
                best_volume = vol
                best_results = results
    
    # Final reinsertion pass on best result
    if best_results is not None and time.time() - start_time < DEADLINE - 5:
        left_out = [i for i in range(n) if best_results[i][0] == -1]
        if left_out:
            placed = [(x, y, z, w, h, d) for (b, x, y, z, w, h, d) in best_results if b == 0]
            best_results, placed = try_reinsert(placed, best_results, left_out)
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
# EVOLVE-BLOCK-END
