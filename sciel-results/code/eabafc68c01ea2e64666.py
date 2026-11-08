import time
import itertools

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Helper to check overlap
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1: return False
        if y1 + h1 <= y2 or y2 + h2 <= y1: return False
        if z1 + d1 <= z2 or z2 + d2 <= z1: return False
        return True

    def get_orientations(w, h, d):
        # Unique orientations to avoid redundant checks
        perms = set(itertools.permutations([w, h, d]))
        return list(perms)

    def greedy_pack(sort_key, pos_generator):
        # Sort boxes based on key
        indexed_boxes = sorted(enumerate(boxes), key=lambda x: sort_key(x[1]), reverse=True)
        
        placed = [] # List of (idx, x, y, z, w, h, d)
        best_vol = 0
        
        for idx, (w, h, d) in indexed_boxes:
            orientations = get_orientations(w, h, d)
            best_placement = None
            best_score = float('inf') # We want to minimize x+y+z
            
            # Try each orientation
            for ow, oh, od in orientations:
                # Get candidate positions
                candidate_positions = pos_generator(placed, ow, oh, od)
                
                for cx, cy, cz in candidate_positions:
                    # Check bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check overlap with placed boxes
                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: prefer positions closer to origin (x, y, z)
                    score = cx + cy + cz
                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)
            
            if best_placement:
                placed.append(best_placement)
                best_vol += best_placement[4] * best_placement[5] * best_placement[6]
            else:
                # Box cannot be placed
                pass
        
        # Construct result
        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        
        return result, best_vol

    def corners_generator(placed, ow, oh, od):
        """Generate candidate positions from corners of placed boxes"""
        corners = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            corners.append((px + pw, py, pz))
            corners.append((px, py + ph, pz))
            corners.append((px, py, pz + pd))
        return corners

    def extended_corners_generator(placed, ow, oh, od):
        """Generate more candidate positions including combinations"""
        corners = [(0, 0, 0)]
        for _, px, py, pz, pw, ph, pd in placed:
            corners.append((px + pw, py, pz))
            corners.append((px, py + ph, pz))
            corners.append((px, py, pz + pd))
            # Add some combined positions
            corners.append((px + pw, py + ph, pz))
            corners.append((px + pw, py, pz + pd))
            corners.append((px, py + ph, pz + pd))
        return corners

    # Strategy 1: Sort by Volume, use basic corners
    res1, vol1 = greedy_pack(lambda b: b[0]*b[1]*b[2], corners_generator)
    if time.time() - start_time > DEADLINE:
        return res1

    # Strategy 2: Sort by Max Dimension, use extended corners
    res2, vol2 = greedy_pack(lambda b: max(b), extended_corners_generator)
    if time.time() - start_time > DEADLINE:
        return res2 if vol2 > vol1 else res1

    # Strategy 3: Sort by Sum of Dimensions, use extended corners
    res3, vol3 = greedy_pack(lambda b: sum(b), extended_corners_generator)
    if time.time() - start_time > DEADLINE:
        best_res = res1
        best_vol = vol1
        if vol2 > best_vol:
            best_res = res2
            best_vol = vol2
        if vol3 > best_vol:
            best_res = res3
            best_vol = vol3
        return best_res

    # Strategy 4: Sort by elongation (max_dim / min_dim), use extended corners
    res4, vol4 = greedy_pack(lambda b: max(b) / max(min(b), 1e-9), extended_corners_generator)
    
    best_res = res1
    best_vol = vol1
    if vol2 > best_vol:
        best_res = res2
        best_vol = vol2
    if vol3 > best_vol:
        best_res = res3
        best_vol = vol3
    if vol4 > best_vol:
        best_res = res4
        best_vol = vol4
        
    return best_res
