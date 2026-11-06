"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute all 6 rotations for each box
    def get_rotations(w, h, d):
        # All 6 unique permutations of (w, h, d)
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    def pack_boxes(ordering, placed_list, results_list, deadline_check=None):
        """Try to place boxes in the given ordering. Returns (placed_count, total_volume_placed)."""
        placed = []
        
        for orig_idx, (bw_orig, bh_orig, bd_orig) in ordering:
            if deadline_check and time.time() > deadline_check:
                break
            
            rots = get_rotations(bw_orig, bh_orig, bd_orig)
            best_pos = None
            best_key = None
            
            # Get candidate positions from placed boxes
            xs = {0}
            ys = {0}
            zs = {0}
            for px, py, pz, pw, ph, pd in placed:
                xs.add(px + pw)
                ys.add(py + ph)
                zs.add(pz + pd)
            xs = sorted(xs)
            ys = sorted(ys)
            zs = sorted(zs)
            
            for (rw, rh, rd) in rots:
                for cx in xs:
                    if cx + rw > W:
                        continue
                    for cy in ys:
                        if cy + rh > H:
                            continue
                        for cz in zs:
                            if cz + rd > D:
                                continue
                            # Check overlap
                            overlap = False
                            for px, py, pz, pw, ph, pd in placed:
                                if (cx < px + pw and cx + rw > px and
                                    cy < py + ph and cy + rh > py and
                                    cz < pz + pd and cz + rd > pz):
                                    overlap = True
                                    break
                            if overlap:
                                continue
                            # Score: BLF (minimize z, then x, then y)
                            key = (cz, cx, cy)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_pos = (cx, cy, cz, rw, rh, rd)
            
            if best_pos is not None:
                cx, cy, cz, rw, rh, rd = best_pos
                placed.append((cx, cy, cz, rw, rh, rd))
                results_list[orig_idx] = (0, cx, cy, cz, rw, rh, rd)
            else:
                results_list[orig_idx] = (-1, 0, 0, 0, bw_orig, bh_orig, bd_orig)
        
        # Fill in any remaining
        for i in range(n):
            if results_list[i] is None:
                results_list[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Calculate score
        total_vol = sum(r[3]*r[4]*r[5] for r in results_list if r[0] == 0)
        placed_count = sum(1 for r in results_list if r[0] == 0)
        return placed_count, total_vol
    
    # Strategy 1: Volume descending, then max dimension
    order1 = list(enumerate(boxes))
    order1.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # Strategy 2: Max dimension descending, then volume
    order2 = list(enumerate(boxes))
    order2.sort(key=lambda x: (-max(x[1]), -(x[1][0]*x[1][1]*x[1][2])))
    
    # Strategy 3: Surface area descending
    order3 = list(enumerate(boxes))
    order3.sort(key=lambda x: (-(2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][0]*x[1][2])), -max(x[1])))
    
    best_result = None
    best_score = 0
    
    for strategy_idx, ordering in enumerate([order1, order2, order3]):
        if time.time() > deadline:
            break
        
        results = [None] * n
        placed_count, total_vol = pack_boxes(ordering, None, results, deadline)
        
        container_vol = W * H * D
        score = total_vol / container_vol if container_vol > 0 else 0
        
        if score > best_score:
            best_score = score
            best_result = list(results)
    
    # If no result yet, use first strategy
    if best_result is None:
        results = [None] * n
        pack_boxes(order1, None, results, deadline)
        best_result = results
    
    return best_result
# EVOLVE-BLOCK-END
