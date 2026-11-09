import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 110s.
    DEADLINE = start_time + 110.0
    
    W, H, D = container
    n_boxes = len(boxes)
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        return list(set(itertools.permutations([w, h, d])))
    
    box_orientations = [get_orientations(w, h, d) for (w, h, d) in boxes]

    # Determine if we can use a 3D grid for fast overlap checks
    # This is significantly faster than O(N) list checks for dense packings
    MAX_GRID_VOLUME = 100000  # 100k cells is small for NumPy
    use_grid = (W * H * D <= MAX_GRID_VOLUME)
    
    if use_grid:
        grid = np.zeros((W, H, D), dtype=np.int8)
    else:
        grid = None

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    def is_valid_grid(x, y, z, w, h, d):
        """Check if a box fits in the grid without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any cell in the volume is already occupied
        if np.any(grid[x:x+w, y:y+h, z:z+d] > 0):
            return False
        return True

    def update_grid(x, y, z, w, h, d, val=1):
        grid[x:x+w, y:y+h, z:z+d] = val

    def is_valid_list(placed, x, y, z, w, h, d):
        """Check if a box fits in the list of placed boxes without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for _, px, py, pz, pw, ph, pd in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def get_candidates(placed, ow, oh, od, strategy='corners'):
        """
        Generate candidate positions.
        'corners': Standard 7-corner expansion. Bounded by 7*N.
        'skyline': Simulates a 2D skyline on the floor and walls.
        """
        candidates = set([(0, 0, 0)])
        
        if not placed:
            return list(candidates)

        if strategy == 'corners':
            for _, px, py, pz, pw, ph, pd in placed:
                # 7 corners relative to the placed box
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
        elif strategy == 'mixed':
            # Mix of corners and some random offsets for diversity
            for _, px, py, pz, pw, ph, pd in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                # Add a few random nearby spots
                if random.random() < 0.3:
                    candidates.add((random.randint(0, max(0, W-ow)), 
                                    random.randint(0, max(0, H-oh)), 
                                    random.randint(0, max(0, D-od))))
        else:
            # Fallback to corners
            for _, px, py, pz, pw, ph, pd in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))

        # Filter out-of-bounds
        valid_candidates = []
        for cx, cy, cz in candidates:
            if 0 <= cx <= W - ow and 0 <= cy <= H - oh and 0 <= cz <= D - od:
                valid_candidates.append((cx, cy, cz))
        
        # Limit candidates to prevent explosion in large N scenarios
        if len(valid_candidates) > 200:
            # Sample if too many
            if strategy == 'mixed' and 'random' in dir():
                 pass # already randomized
            else:
                 valid_candidates = valid_candidates[:200]
                 
        return valid_candidates

    def score_position(cx, cy, cz, mode='blb'):
        if mode == 'blb':
            return cz * 1000000 + cy * 1000 + cx
        elif mode == 'bottom_left':
            return cz * 1000000 + cx * 1000 + cy
        elif mode == 'left_bottom':
            return cy * 1000000 + cz * 1000 + cx
        else:
            return cx + cy + cz

    def greedy_pack(sort_key, gen_strategy, score_mode, rng=None, perturbation=0.0):
        indexed_boxes = sorted(range(n_boxes), key=lambda i: sort_key(boxes[i]), reverse=True)
        
        placed = []
        total_vol = 0
        
        for idx in indexed_boxes:
            if time.time() > DEADLINE:
                break
            
            best_placement = None
            best_score = float('inf')
            
            for ow, oh, od in box_orientations[idx]:
                candidates = get_candidates(placed, ow, oh, od, strategy=gen_strategy)
                
                for cx, cy, cz in candidates:
                    if use_grid:
                        if not is_valid_grid(cx, cy, cz, ow, oh, od):
                            continue
                    else:
                        if not is_valid_list(placed, cx, cy, cz, ow, oh, od):
                            continue
                    
                    s = score_position(cx, cy, cz, mode=score_mode)
                    if perturbation > 0 and rng is not None:
                        s += rng.random() * perturbation
                        
                    if s < best_score:
                        best_score = s
                        best_placement = (idx, cx, cy, cz, ow, oh, od)
            
            if best_placement is not None:
                idx_p, bx, by, bz, bw, bh, bd = best_placement
                placed.append(best_placement)
                total_vol += bw * bh * bd
                
                if use_grid:
                    update_grid(bx, by, bz, bw, bh, bd, 1)

        # Phase 2: Insertion pass for unplaced boxes (simplified for speed)
        placed_indices = {p[0] for p in placed}
        unplaced = [i for i in range(n_boxes) if i not in placed_indices]
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        for idx in unplaced:
            if time.time() > DEADLINE:
                break
            
            best_placement = None
            best_score = float('inf')
            
            for ow, oh, od in box_orientations[idx]:
                # Use a denser set of candidates for insertion, but still bounded
                # We try all unique x,y,z from placed boxes plus 0
                xs = {0}
                ys = {0}
                zs = {0}
                for _, px, py, pz, pw, ph, pd in placed:
                    xs.add(px)
                    xs.add(px + pw)
                    ys.add(py)
                    ys.add(py + ph)
                    zs.add(pz)
                    zs.add(pz + pd)
                
                # Filter bounds
                xs = {x for x in xs if 0 <= x <= W - ow}
                ys = {y for y in ys if 0 <= y <= H - oh}
                zs = {z for z in zs if 0 <= z <= D - od}
                
                # Limit combinations
                if len(xs) * len(ys) * len(zs) > 500:
                    # Sample
                    if rng:
                        xs_sample = rng.sample(list(xs), min(len(xs), 10))
                        ys_sample = rng.sample(list(ys), min(len(ys), 10))
                        zs_sample = rng.sample(list(zs), min(len(zs), 10))
                    else:
                        xs_sample = list(xs)[:10]
                        ys_sample = list(ys)[:10]
                        zs_sample = list(zs)[:10]
                    candidates = list(itertools.product(xs_sample, ys_sample, zs_sample))
                else:
                    candidates = list(itertools.product(xs, ys, zs))

                for cx, cy, cz in candidates:
                    if use_grid:
                        if not is_valid_grid(cx, cy, cz, ow, oh, od):
                            continue
                    else:
                        if not is_valid_list(placed, cx, cy, cz, ow, oh, od):
                            continue
                    
                    s = score_position(cx, cy, cz, mode='simple')
                    
                    if s < best_score:
                        best_score = s
                        best_placement = (idx, cx, cy, cz, ow, oh, od)
            
            if best_placement is not None:
                idx_p, bx, by, bz, bw, bh, bd = best_placement
                placed.append(best_placement)
                total_vol += bw * bh * bd
                
                if use_grid:
                    update_grid(bx, by, bz, bw, bh, bd, 1)

        result = [(-1, 0, 0, 0, 0, 0, 0)] * n_boxes
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        return result, total_vol

    # --- Sorting Strategies ---
    sort_strategies = [
        lambda b: b[0]*b[1]*b[2],           # Volume
        lambda b: max(b),                     # Max dimension
        lambda b: sum(b),                     # Sum of dimensions
    ]

    # --- Configuration ---
    # Reduced set of configurations to ensure we finish in time
    configs = [
        ('corners', 'blb'),
        ('corners', 'bottom_left'),
        ('mixed', 'blb'),
        ('mixed', 'simple'),
    ]

    best_result = None
    best_vol = 0

    # Phase 1: Deterministic Strategies
    for sort_key in sort_strategies:
        if time.time() > DEADLINE:
            break
        for gen_str, score_mod in configs:
            if time.time() > DEADLINE:
                break
            res, vol = greedy_pack(sort_key, gen_str, score_mod)
            if vol > best_vol:
                best_vol = vol
                best_result = res

    # Phase 2: Randomized Search
    rng = random.Random(42)
    max_restarts = 50
    
    restart_count = 0
    while time.time() < DEADLINE - 1.0 and restart_count < max_restarts:
        restart_count += 1
        
        sort_key = sort_strategies[rng.randint(0, len(sort_strategies) - 1)]
        gen_str = rng.choice(['corners', 'mixed'])
        score_mod = rng.choice(['blb', 'bottom_left', 'simple'])
        perturbation = rng.uniform(0.1, 5.0)
        
        res, vol = greedy_pack(sort_key, gen_str, score_mod, rng=rng, perturbation=perturbation)
        
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Fallback if nothing was placed
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
