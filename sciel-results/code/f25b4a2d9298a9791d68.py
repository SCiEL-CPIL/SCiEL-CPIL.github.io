"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Precompute orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    orientations_cache = [get_orientations(w, h, d) for w, h, d in boxes]
    
    # 3D occupancy grid for fast overlap checking
    # Use a resolution of 1 unit; if container is large, we may need to scale
    # For now, assume integer coordinates and use direct indexing
    occ = np.zeros((W, H, D), dtype=np.uint8)
    
    def mark_occ(x, y, z, w, h, d, val=1):
        """Mark a box region in the occupancy grid."""
        occ[x:x+w, y:y+h, z:z+d] = val
    
    def check_fits(x, y, z, w, h, d):
        """Check if a box fits at position (x,y,z) with dimensions (w,h,d)."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Vectorized overlap check: if any cell in the region is occupied, it doesn't fit
        if np.any(occ[x:x+w, y:y+h, z:z+d]):
            return False
        return True
    
    # Generate candidate positions from current placed boxes
    def get_candidates(placed):
        cands = [(0, 0, 0)]
        seen = set(cands)
        for (px, py, pz, pw, ph, pd) in placed:
            candidates = [
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
                (px + pw, py + ph, pz),
                (px + pw, py, pz + pd),
                (px, py + ph, pz + pd),
                (px + pw, py + ph, pz + pd),
            ]
            for (nx, ny, nz) in candidates:
                if (nx, ny, nz) not in seen and 0 <= nx <= W and 0 <= ny <= H and 0 <= nz <= D:
                    seen.add((nx, ny, nz))
                    cands.append((nx, ny, nz))
        return cands
    
    # Score a position: prefer lower z, then lower y, then lower x
    def position_score(x, y, z, w, h, d):
        return z * W * H + y * W + x
    
    # Try to place a single box given current placed list; return (x,y,z,ow,oh,od) or None
    def place_one(box_idx, placed):
        w, h, d = boxes[box_idx]
        best = None
        best_score = float('inf')
        for (ow, oh, od) in orientations_cache[box_idx]:
            if ow > W or oh > H or od > D:
                continue
            for (ax, ay, az) in get_candidates(placed):
                if check_fits(ax, ay, az, ow, oh, od):
                    s = position_score(ax, ay, az, ow, oh, od)
                    if s < best_score:
                        best_score = s
                        best = (ax, ay, az, ow, oh, od)
        return best
    
    def evaluate(result):
        vol = 0
        for p in result:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        return vol / container_vol
    
    # Greedy packing with a given order
    def greedy(order):
        placed = []
        result = [None] * n
        for idx in order:
            pos = place_one(idx, placed)
            if pos is not None:
                x, y, z, ow, oh, od = pos
                placed.append((x, y, z, ow, oh, od))
                mark_occ(x, y, z, ow, oh, od)
                result[idx] = (0, x, y, z, ow, oh, od)
            else:
                w, h, d = boxes[idx]
                result[idx] = (-1, 0, 0, 0, w, h, d)
        return result
    
    # Reset occupancy grid
    def reset_occ():
        occ[:] = 0
    
    # Build various orders
    indices = list(range(n))
    
    def order_by_volume():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def order_by_maxdim():
        return sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    
    def order_by_mindim():
        return sorted(indices, key=lambda i: min(boxes[i]), reverse=True)
    
    def order_by_surface():
        return sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1]+boxes[i][1]*boxes[i][2]+boxes[i][2]*boxes[i][0]), reverse=True)
    
    def order_by_aspect():
        def asp(i):
            w,h,d = boxes[i]
            return max(w,h,d)/min(w,h,d) if min(w,h,d)>0 else 0
        return sorted(indices, key=lambda i: asp(i))
    
    def order_by_min_volume():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    def order_by_maxdim_min():
        return sorted(indices, key=lambda i: max(boxes[i]) + min(boxes[i]), reverse=True)
    
    def order_random():
        r = list(indices)
        random.shuffle(r)
        return r
    
    # Local search: re-insert left-out boxes
    def local_improve(result, budget):
        """Try to improve by re-inserting left-out boxes."""
        improved = True
        while improved and time.time() < budget:
            improved = False
            left_out = [i for i in range(n) if result[i][0] == -1]
            left_out.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
            for idx in left_out:
                if time.time() >= budget:
                    break
                # Rebuild placed list from result
                placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in result if p[0]==0]
                pos = place_one(idx, placed)
                if pos is not None:
                    x, y, z, ow, oh, od = pos
                    mark_occ(x, y, z, ow, oh, od)
                    result[idx] = (0, x, y, z, ow, oh, od)
                    improved = True
        return result
    
    # Swap improvement: remove a placed box, try to place a left-out box, then re-place removed box
    def swap_improve(result, budget):
        """Try swapping to improve packing."""
        left_out = [i for i in range(n) if result[i][0] == -1]
        if not left_out:
            return result
        
        left_out.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        placed_indices = [i for i in range(n) if result[i][0] == 0]
        
        for lo_idx in left_out[:10]:
            if time.time() >= budget:
                break
            lo_vol = boxes[lo_idx][0] * boxes[lo_idx][1] * boxes[lo_idx][2]
            for pi_idx in placed_indices:
                if time.time() >= budget:
                    break
                pi_vol = boxes[pi_idx][0] * boxes[pi_idx][1] * boxes[pi_idx][2]
                if lo_vol < pi_vol * 0.8:
                    continue
                
                saved_pos = result[pi_idx]
                # Unmark the placed box
                mark_occ(saved_pos[1], saved_pos[2], saved_pos[3], saved_pos[4], saved_pos[5], saved_pos[6], val=0)
                
                placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in result if p[0]==0 and p is not saved_pos]
                
                # Try placing lo_idx
                pos_lo = place_one(lo_idx, placed)
                if pos_lo is not None:
                    mark_occ(pos_lo[0], pos_lo[1], pos_lo[2], pos_lo[3], pos_lo[4], pos_lo[5])
                    placed_new = placed + [pos_lo]
                    
                    # Try to re-place pi_idx
                    pos_pi = place_one(pi_idx, placed_new)
                    if pos_pi is not None:
                        mark_occ(pos_pi[0], pos_pi[1], pos_pi[2], pos_pi[3], pos_pi[4], pos_pi[5])
                        
                        result_new = list(result)
                        result_new[lo_idx] = (0, pos_lo[0], pos_lo[1], pos_lo[2], pos_lo[3], pos_lo[4], pos_lo[5])
                        result_new[pi_idx] = (0, pos_pi[0], pos_pi[1], pos_pi[2], pos_pi[3], pos_pi[4], pos_pi[5])
                        
                        if evaluate(result_new) > evaluate(result):
                            return result_new
                        else:
                            # Revert
                            mark_occ(pos_lo[0], pos_lo[1], pos_lo[2], pos_lo[3], pos_lo[4], pos_lo[5], val=0)
                            mark_occ(pos_pi[0], pos_pi[1], pos_pi[2], pos_pi[3], pos_pi[4], pos_pi[5], val=0)
                            mark_occ(saved_pos[1], saved_pos[2], saved_pos[3], saved_pos[4], saved_pos[5], saved_pos[6], val=1)
                else:
                    # Revert
                    mark_occ(saved_pos[1], saved_pos[2], saved_pos[3], saved_pos[4], saved_pos[5], saved_pos[6], val=1)
        return result
    
    best_result = None
    best_fill = 0
    
    # Try deterministic strategies
    strategies = [order_by_volume, order_by_maxdim, order_by_mindim, order_by_surface, 
                  order_by_aspect, order_by_min_volume, order_by_maxdim_min]
    
    for strat in strategies:
        if time.time() > deadline - 5:
            break
        reset_occ()
        order = strat()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = list(res)
            # Local improvement
            if time.time() < deadline - 10:
                res = local_improve(res, deadline - 10)
                fill2 = evaluate(res)
                if fill2 > best_fill:
                    best_fill = fill2
                    best_result = list(res)
                # Try swap improvement
                if time.time() < deadline - 5:
                    res = swap_improve(res, deadline - 5)
                    fill3 = evaluate(res)
                    if fill3 > best_fill:
                        best_fill = fill3
                        best_result = list(res)
    
    # Try random orders with local improvement
    n_random = 40
    for r in range(n_random):
        if time.time() > deadline - 3:
            break
        reset_occ()
        order = order_random()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = list(res)
            if time.time() < deadline - 8:
                res = local_improve(res, deadline - 8)
                fill2 = evaluate(res)
                if fill2 > best_fill:
                    best_fill = fill2
                    best_result = list(res)
                if time.time() < deadline - 4:
                    res = swap_improve(res, deadline - 4)
                    fill3 = evaluate(res)
                    if fill3 > best_fill:
                        best_fill = fill3
                        best_result = list(res)
    
    # Final: try to re-insert any remaining left-out boxes into the best result
    if best_result is not None and time.time() < deadline - 1:
        reset_occ()
        # Rebuild occupancy from best_result
        for p in best_result:
            if p[0] == 0:
                mark_occ(p[1], p[2], p[3], p[4], p[5], p[6])
        for idx in range(n):
            if best_result[idx][0] == -1:
                placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in best_result if p[0]==0]
                pos = place_one(idx, placed)
                if pos is not None:
                    x,y,z,ow,oh,od = pos
                    mark_occ(x,y,z,ow,oh,od)
                    best_result[idx] = (0,x,y,z,ow,oh,od)
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
# EVOLVE-BLOCK-END
