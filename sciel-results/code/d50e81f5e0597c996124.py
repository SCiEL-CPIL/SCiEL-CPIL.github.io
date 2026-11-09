import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 145s.
    DEADLINE = start_time + 145.0
    
    W, H, D = container
    n_boxes = len(boxes)
    
    # Precompute unique orientations for each box
    # This saves time in the inner loop
    def get_orientations(w, h, d):
        return list(set(itertools.permutations([w, h, d])))
    
    box_orientations = [get_orientations(w, h, d) for (w, h, d) in boxes]

    # Determine if we can use a height map for fast pre-filtering
    # If dimensions are small enough, a 2D grid is fast.
    # If too large, we fall back to pure list-based overlap checks.
    MAX_GRID_SIZE = 250
    use_height_map = (W <= MAX_GRID_SIZE and H <= MAX_GRID_SIZE)
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    # --- Position Generators ---
    # These take the list of placed boxes and generate candidate (x,y,z) positions.
    
    def corners_generator(placed, ow, oh, od):
        """Standard 7-corner expansion from each placed box."""
        candidates = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        return candidates

    def shelf_generator(placed, ow, oh, od):
        """Shelf-based: Focus on stacking on top and sides to create 'shelves'."""
        candidates = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            # Top of box
            candidates.append((px, py, pz + pd))
            # Right side
            candidates.append((px + pw, py, pz))
            # Front side
            candidates.append((px, py + ph, pz))
            # Corners of the shelf
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        return candidates

    def extended_generator(placed, ow, oh, od):
        """Combines corners and shelf logic, deduplicated."""
        candidates = set([(0, 0, 0)])
        for _, px, py, pz, pw, ph, pd in placed:
            # All 8 corners of the placed box
            candidates.add((px, py, pz))
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        return list(candidates)

    # --- Scoring Functions ---
    # Lower score is better. We prioritize bottom (z), then left (x), then back (y)
    # but with different weights to explore different packing styles.
    
    def score_blb(cx, cy, cz):
        return cz * 1000000 + cy * 1000 + cx

    def score_bottom_left(cx, cy, cz):
        return cz * 1000000 + cx * 1000 + cy

    def score_left_bottom(cx, cy, cz):
        return cy * 1000000 + cz * 1000 + cx

    def score_simple(cx, cy, cz):
        return cx + cy + cz

    def greedy_pack(sort_key, pos_generator, score_func, rng=None, perturbation=0.0):
        """
        Core greedy packing algorithm.
        sort_key: function to sort boxes
        pos_generator: function to get candidate positions
        score_func: function to rank positions
        rng: random number generator for perturbation
        perturbation: amount of random noise added to score
        """
        indexed_boxes = sorted(range(n_boxes), key=lambda i: sort_key(boxes[i]), reverse=True)
        
        placed = []  # List of (idx, x, y, z, w, h, d)
        total_vol = 0
        
        if use_height_map:
            height_map = np.zeros((W, H), dtype=np.int32)
        else:
            height_map = None

        for idx in indexed_boxes:
            if time.time() > DEADLINE:
                break
            
            best_placement = None
            best_score = float('inf')
            
            for ow, oh, od in box_orientations[idx]:
                # Generate candidates
                candidates = pos_generator(placed, ow, oh, od)
                
                # Optional: Limit candidates if too many to save time
                # In practice, with 60 boxes, this list stays manageable (< 500)
                
                for cx, cy, cz in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Fast pre-filter using height map if available
                    if height_map is not None:
                        # Check if the footprint is clear enough height-wise
                        # Note: This is a necessary but not sufficient condition for overlap
                        # We still need the exact check below.
                        if cx + ow <= W and cy + oh <= H:
                            footprint = height_map[cx:cx+ow, cy:cy+oh]
                            if footprint.size > 0:
                                max_h = int(np.max(footprint))
                                # If the existing height is higher than where we want to place,
                                # AND that existing block is actually in the way (same z range), it's bad.
                                # However, simply checking max_h > cz + od is not quite right for all cases,
                                # but if max_h > D, it's definitely bad.
                                # A safer fast check: if any cell in footprint has height > cz, 
                                # it might overlap. But height map stores max height per (x,y).
                                # If max_h > cz, there is a box above cz at that (x,y). 
                                # If that box starts below cz+od, it overlaps.
                                # Since we don't know start z, we just do the exact check if this passes.
                                # Actually, if max_h > cz + od, it's definitely a conflict? No.
                                # If max_h > cz, it *might* be a conflict.
                                # Let's just rely on the exact check for correctness, 
                                # but use the map to skip obvious "too high" placements?
                                # If cz >= max_h, it's safe from vertical overlap with existing blocks 
                                # that end below cz. But if there's a block from z=0 to z=10, and we place at z=5, 
                                # max_h=10 > 5, so we don't skip. Correct.
                                # If we place at z=10, max_h=10 is not > 10, so we don't skip. Correct.
                                # This pre-filter is weak but safe.
                                pass 
                    
                    # Exact overlap check
                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Calculate score
                    s = score_func(cx, cy, cz)
                    if perturbation > 0 and rng is not None:
                        s += rng.random() * perturbation
                        
                    if s < best_score:
                        best_score = s
                        best_placement = (idx, cx, cy, cz, ow, oh, od)
            
            if best_placement is not None:
                idx_p, bx, by, bz, bw, bh, bd = best_placement
                placed.append(best_placement)
                total_vol += bw * bh * bd
                
                # Update height map
                if height_map is not None:
                    height_map[bx:bx+bw, by:by+bh] = np.maximum(
                        height_map[bx:bx+bw, by:by+bh], bz + bd)

        # Construct result
        result = [(-1, 0, 0, 0, 0, 0, 0)] * n_boxes
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        return result, total_vol

    # --- Sorting Strategies ---
    sort_strategies = [
        lambda b: b[0]*b[1]*b[2],           # Volume
        lambda b: max(b),                     # Max dimension
        lambda b: sum(b),                     # Sum of dimensions
        lambda b: max(b) / max(min(b), 1e-9),# Elongation
        lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]),  # Surface area
        lambda b: b[0]*b[1]*b[2] / max(b),   # Volume per max dim
    ]

    # --- Position Generators & Score Functions ---
    pos_generators = [corners_generator, shelf_generator, extended_generator]
    score_functions = [score_blb, score_bottom_left, score_left_bottom, score_simple]

    best_result = None
    best_vol = 0

    # Phase 1: Deterministic Strategies
    # Try all combinations of sorting, position generation, and scoring
    for sort_key in sort_strategies:
        if time.time() > DEADLINE:
            break
        for pos_gen in pos_generators:
            if time.time() > DEADLINE:
                break
            for score_func in score_functions:
                if time.time() > DEADLINE:
                    break
                res, vol = greedy_pack(sort_key, pos_gen, score_func)
                if vol > best_vol:
                    best_vol = vol
                    best_result = res

    # Phase 2: Randomized Search
    # Use random restarts with perturbation to escape local optima
    rng = random.Random(42)
    max_restarts = 100  # Cap to prevent infinite loop if time is generous
    
    restart_count = 0
    while time.time() < DEADLINE - 2.0 and restart_count < max_restarts:
        restart_count += 1
        
        # Randomly select components
        sort_key = sort_strategies[rng.randint(0, len(sort_strategies) - 1)]
        pos_gen = pos_generators[rng.randint(0, len(pos_generators) - 1)]
        score_func = score_functions[rng.randint(0, len(score_functions) - 1)]
        
        # Add some perturbation to encourage diversity
        perturbation = rng.uniform(0.1, 2.0)
        
        res, vol = greedy_pack(sort_key, pos_gen, score_func, rng=rng, perturbation=perturbation)
        
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Fallback if nothing was placed
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result

def _load_fallback(boxes, container, deadline):
    # This function is kept for interface compatibility but the main load()
    # handles large containers by simply not using the height map.
    # If we ever need a specific fallback, we can call load() again or
    # implement a simpler list-only version here.
    # For now, we just call the main load logic which adapts via use_height_map.
    return load(boxes, container)
