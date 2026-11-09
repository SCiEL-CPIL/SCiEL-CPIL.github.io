"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools

def load(boxes, container):
    W, H, D = container
    n_boxes = len(boxes)
    
    # Helper to check overlap
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
    
    def pack_with_order(order):
        placed = []
        results = [None] * n_boxes
        
        # Precompute orientations for each box to avoid recalculating
        # But we need to access by index.
        
        def get_candidates():
            candidates = [(0, 0, 0)]
            for (px, py, pz, pw, ph, pd) in placed:
                # Only add if within bounds to reduce list size
                if px + pw < W: candidates.append((px + pw, py, pz))
                if py + ph < H: candidates.append((px, py + ph, pz))
                if pz + pd < D: candidates.append((px, py, pz + pd))
                
                if px + pw < W and py + ph < H: candidates.append((px + pw, py + ph, pz))
                if px + pw < W and pz + pd < D: candidates.append((px + pw, py, pz + pd))
                if py + ph < H and pz + pd < D: candidates.append((px, py + ph, pz + pd))
                if px + pw < W and py + ph < H and pz + pd < D: candidates.append((px + pw, py + ph, pz + pd))
            
            # Deduplicate
            unique_candidates = list(set(candidates))
            return unique_candidates
        
        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for (px, py, pz, pw, ph, pd) in placed:
                if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                    return False
            return True

        total_volume = 0
        for orig_idx in order:
            w_orig, h_orig, d_orig = boxes[orig_idx]
            orientations = list(set(itertools.permutations((w_orig, h_orig, d_orig))))
            
            best_pos = None
            best_score = -1
            
            # Get candidates once per box, not per orientation, as they don't depend on orientation
            # Wait, candidates are positions (x,y,z). The dimensions (w,h,d) change.
            # But the set of candidate *positions* is independent of the box size.
            # So we can compute candidates once.
            candidates = get_candidates()
            
            for ow, oh, od in orientations:
                for (cx, cy, cz) in candidates:
                    if can_place(cx, cy, cz, ow, oh, od):
                        # Heuristic: Minimize z, then y, then x.
                        # This tends to fill the bottom and front.
                        score = -(cz * (W * H) + cy * W + cx)
                        if score > best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, ow, oh, od)
            
            if best_pos is not None:
                x, y, z, ow, oh, od = best_pos
                placed.append((x, y, z, ow, oh, od))
                results[orig_idx] = (0, x, y, z, ow, oh, od)
                total_volume += ow * oh * od
            else:
                results[orig_idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        return results, total_volume

    # Define multiple sorting strategies
    indices = list(range(n_boxes))
    
    strategies = []
    
    # 1. Volume descending
    strategies.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    strategies.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Surface area descending
    strategies.append(sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 4. Aspect ratio (max/min) descending
    strategies.append(sorted(indices, key=lambda i: max(boxes[i]) / min(boxes[i]), reverse=True))
    
    # 5. Volume ascending (sometimes helps?)
    strategies.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]))

    best_result = None
    best_volume = -1
    
    for strategy in strategies:
        res, vol = pack_with_order(strategy)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    return best_result

# EVOLVE-BLOCK-END
