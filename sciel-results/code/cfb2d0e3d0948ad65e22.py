"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0
    
    # Precompute unique rotations for each box
    rotations = []
    for w, h, d in boxes:
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        rotations.append(rots)
    
    def solve_order(order):
        """Try to place boxes in the given order using best-fit decreasing heuristic."""
        placed = []  # list of (x, y, z, w, h, d)
        result = [None] * n  # result[i] = placement tuple or None
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            best_pos = None
            best_score = None
            
            for (rw, rh, rd) in rotations[idx]:
                # Try to find best position
                # Use a simple grid-based search with adaptive step
                # For each candidate x, y, z, check if it fits
                
                # We'll use a smarter approach: try to place at the lowest possible z,
                # then smallest x, then smallest y (or some heuristic)
                
                # First, check if box fits in container
                if rw > W or rh > H or rd > D:
                    continue
                
                # Try candidate positions
                # We'll sample positions: corners and edges of existing boxes
                candidates = [(0, 0, 0)]
                
                for (px, py, pz, pw, ph, pd) in placed:
                    # Positions relative to existing boxes
                    candidates.append((px + pw, py, pz))
                    candidates.append((px, py + ph, pz))
                    candidates.append((px, py, pz + pd))
                    # Also try combinations
                    candidates.append((px + pw, py + ph, pz))
                    candidates.append((px + pw, py, pz + pd))
                    candidates.append((px, py + ph, pz + pd))
                    candidates.append((px + pw, py + ph, pz + pd))
                
                # Deduplicate and filter
                seen_pos = set()
                valid_candidates = []
                for (cx, cy, cz) in candidates:
                    if cx < 0 or cy < 0 or cz < 0:
                        continue
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    key = (cx, cy, cz)
                    if key in seen_pos:
                        continue
                    seen_pos.add(key)
                    
                    # Check overlap with all placed boxes
                    fits = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (cx + rw <= px or px + pw <= cx):
                            if not (cy + rh <= py or py + ph <= cy):
                                if not (cz + rd <= pz or pz + pd <= cz):
                                    fits = False
                                    break
                    if fits:
                        valid_candidates.append((cx, cy, cz))
                
                if not valid_candidates:
                    continue
                
                # Score each candidate: prefer lower z, then smaller x, then smaller y
                # Also consider how "tight" the fit is
                for (cx, cy, cz) in valid_candidates:
                    # Score: lower is better
                    score = (cz * 1000000 + cx * 1000 + cy)
                    # Also add a small penalty for gaps
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, rw, rh, rd)
            
            if best_pos is not None:
                placed.append(best_pos)
                result[idx] = (0, best_pos[0], best_pos[1], best_pos[2], best_pos[3], best_pos[4], best_pos[5])
            else:
                result[idx] = None
        
        return result, placed
    
    # Generate several candidate orderings
    # 1. Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # 2. Max dimension descending
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    
    # 3. Volume ascending (sometimes helps)
    vol_asc_order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
    
    # 4. Random permutations
    import random
    random.seed(42)
    random_orders = []
    for _ in range(10):
        order = list(range(n))
        random.shuffle(order)
        random_orders.append(order)
    
    all_orders = [vol_order, maxdim_order, vol_asc_order] + random_orders
    
    best_result = None
    best_vol = -1
    
    for order in all_orders:
        if time.time() > deadline:
            break
        result, placed = solve_order(order)
        total_vol = sum(p[3] * p[4] * p[5] for p in placed)
        if total_vol > best_vol:
            best_vol = total_vol
            best_result = result
        # Early stop if we've filled a lot
        if total_vol >= 0.9 * W * H * D:
            break
    
    # If no result, return all left out
    if best_result is None:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Fill in missing entries
    final = []
    for i in range(n):
        if best_result[i] is not None:
            final.append(best_result[i])
        else:
            w, h, d = boxes[i]
            final.append((-1, 0, 0, 0, w, h, d))
    
    return final

# EVOLVE-BLOCK-END
