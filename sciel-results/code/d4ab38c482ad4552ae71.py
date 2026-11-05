"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 118.0

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    def get_orientations(w, h, d):
        """Return unique orientations (permutations of dims)."""
        dims = (w, h, d)
        seen = set()
        orients = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        return orients

    def run_strategy(order_key_func, placement_metric='best_fit', seed=None):
        """
        Runs a packing strategy.
        placement_metric: 'first_fit', 'best_fit', 'best_fit_volume'
        Returns (total_volume, results_list)
        """
        if time.time() > deadline:
            return 0, None

        indexed_boxes = list(enumerate(boxes))
        try:
            if seed is not None:
                # Shuffle boxes with same volume for randomized variants
                # Group by volume, shuffle within groups
                from collections import defaultdict
                vol_groups = defaultdict(list)
                for idx in indexed_boxes:
                    w, h, d = idx[1]
                    vol = w * h * d
                    vol_groups[vol].append(idx)
                
                shuffled = []
                for vol in sorted(vol_groups.keys(), reverse=True):
                    group = vol_groups[vol]
                    random.Random(seed).shuffle(group)
                    shuffled.extend(group)
                indexed_boxes = shuffled
            else:
                indexed_boxes.sort(key=order_key_func)
        except Exception:
            return 0, None

        placed = []  # List of (x, y, z, w, h, d)
        results = [None] * len(boxes)
        total_vol = 0

        # Candidate points for placement
        candidate_points = set([(0, 0, 0)])

        for orig_idx, (w, h, d) in indexed_boxes:
            if time.time() > deadline:
                break

            orientations = get_orientations(w, h, d)

            best_pos = None
            best_score = float('inf')
            best_orient = None
            found_placement = False

            # Sort candidate points for deterministic tie-breaking
            # We'll collect valid placements and pick the best
            valid_placements = []

            for cx, cy, cz in candidate_points:
                if time.time() > deadline:
                    break

                for ow, oh, od in orientations:
                    # Check bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue

                    # Check overlap with all placed boxes
                    overlap = False
                    for px, py, pz, pw, ph, pd in placed:
                        if not (cx + ow <= px or px + pw <= cx or
                                cy + oh <= py or py + ph <= cy or
                                cz + od <= pz or pz + pd <= cz):
                            overlap = True
                            break

                    if overlap:
                        continue

                    # This is a valid placement
                    if placement_metric == 'first_fit':
                        # First fit: just take the first valid one
                        # But we still need to check all to be sure
                        # Actually for first fit, we can break early if we want
                        # But let's be consistent and score it
                        # For first fit, score is just the order of discovery
                        # To make it deterministic, we use position as score
                        score = (cx, cy, cz)
                    elif placement_metric == 'best_fit':
                        # Best fit: minimize the sum of distances to other boxes
                        # Approximate by minimizing cx + cy + cz (closest to origin)
                        # Better: minimize the "wasted space" around the box
                        # Use sum of normalized distances to origin
                        nx = cx / W if W > 0 else 0
                        ny = cy / H if H > 0 else 0
                        nz = cz / D if D > 0 else 0
                        score = nx + ny + nz
                    elif placement_metric == 'best_fit_volume':
                        # Best fit: minimize the bounding box volume of all placed boxes
                        # After placing, what's the max extent?
                        max_x = max([p[0] + p[3] for p in placed] + [cx + ow]) if placed else (cx + ow)
                        max_y = max([p[1] + p[4] for p in placed] + [cy + oh]) if placed else (cy + oh)
                        max_z = max([p[2] + p[5] for p in placed] + [cz + od]) if placed else (cz + od)
                        bbox_vol = max_x * max_y * max_z
                        score = bbox_vol
                    else:
                        score = cx + cy + cz

                    valid_placements.append((score, (cx, cy, cz), (ow, oh, od)))

            if valid_placements:
                # Pick the best placement
                valid_placements.sort(key=lambda x: x[0])
                best_score_val, best_pos, best_orient = valid_placements[0]

                bx, by, bz = best_pos
                bw, bh, bd = best_orient

                placed.append((bx, by, bz, bw, bh, bd))
                results[orig_idx] = (0, bx, by, bz, bw, bh, bd)
                total_vol += bw * bh * bd

                # Generate new candidate positions
                new_candidates = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    # Also try combinations for better gap filling
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]

                for nc in new_candidates:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        candidate_points.add(nc)
            else:
                results[orig_idx] = (-1, 0, 0, 0, w, h, d)

        # Fill any remaining results
        for i in range(len(boxes)):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)

        return total_vol, results

    # Define sorting strategies
    def sort_vol_desc(x):
        w, h, d = x[1]
        return -(w * h * d)

    def sort_max_dim_desc(x):
        w, h, d = x[1]
        return -max(w, h, d)

    def sort_sum_desc(x):
        w, h, d = x[1]
        return -(w + h + d)

    def sort_aspect(x):
        w, h, d = x[1]
        vol = w * h * d
        if vol == 0:
            return (0, 0)
        s = (w + h + d) / 3.0
        dev = abs(w - s) + abs(h - s) + abs(d - s)
        return (-vol, dev)

    best_vol = 0
    best_results = None

    # Run multiple strategies with different placement metrics
    strategies = [
        (sort_vol_desc, 'best_fit'),
        (sort_vol_desc, 'best_fit_volume'),
        (sort_max_dim_desc, 'best_fit'),
        (sort_sum_desc, 'best_fit'),
        (sort_aspect, 'best_fit'),
    ]

    for i, (strat, metric) in enumerate(strategies):
        if time.time() > deadline:
            break

        vol, res = run_strategy(strat, metric)

        if res is not None and vol > best_vol:
            best_vol = vol
            best_results = res

        # Try randomized variants if time permits
        if time.time() < deadline - 15:
            for seed in [42, 123, 456]:
                if time.time() > deadline:
                    break
                vol, res = run_strategy(strat, metric, seed=seed)
                if res is not None and vol > best_vol:
                    best_vol = vol
                    best_results = res

    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
# EVOLVE-BLOCK-END
