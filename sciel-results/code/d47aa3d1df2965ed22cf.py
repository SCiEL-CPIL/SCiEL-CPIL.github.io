"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Preprocess: for each box, generate all valid orientations
    # An orientation is (w, h, d) such that w<=W, h<=H, d<=D
    # We'll keep all 6 permutations but filter by container bounds
    def get_orientations(w, h, d):
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            if perm[0] <= W and perm[1] <= H and perm[2] <= D:
                orients.add(perm)
        if not orients:
            # If no orientation fits, we can't place it
            return [(w, h, d)]  # Return original, will fail placement
        return list(orients)
    
    # Sort boxes by volume descending (BFD heuristic)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    def try_pack(box_order, seed_offset=0):
        """Try to pack boxes in the given order using Best-Fit-Decreasing with corner selection.
        Returns list of placements (bin, x, y, z, w, h, d) for all boxes."""
        placements = [None] * len(boxes)
        placed = []  # list of (x, y, z, w, h, d)
        
        def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
            return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                        y1 + h1 <= y2 or y2 + h2 <= y1 or
                        z1 + d1 <= z2 or z2 + d2 <= z1)
        
        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for px, py, pz, pw, ph, pd in placed:
                if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                    return False
            return True
        
        def find_best_position(w, h, d):
            """Find the position that minimizes (x, y, z) lexicographically among valid positions.
            We search candidate positions based on existing placements and container boundaries."""
            # Candidate x, y, z coordinates
            xs = [0]
            ys = [0]
            zs = [0]
            for px, py, pz, pw, ph, pd in placed:
                xs.append(px + pw)
                ys.append(py + ph)
                zs.append(pz + pd)
            
            # Filter candidates that are within bounds
            xs = sorted(set(x for x in xs if x + w <= W))
            ys = sorted(set(y for y in ys if y + h <= H))
            zs = sorted(set(z for z in zs if z + d <= D))
            
            if not xs or not ys or not zs:
                return None
            
            best_pos = None
            best_key = None
            for x in xs:
                for y in ys:
                    for z in zs:
                        if can_place(x, y, z, w, h, d):
                            key = (x, y, z)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_pos = (x, y, z)
            return best_pos
        
        for idx, (orig_idx, w, h, d) in enumerate(box_order):
            if time.time() > deadline:
                break
            
            orients = get_orientations(w, h, d)
            best_orient = None
            best_pos = None
            best_key = None
            
            for ow, oh, od in orients:
                pos = find_best_position(ow, oh, od)
                if pos is not None:
                    key = (pos[0], pos[1], pos[2])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                placed.append((x, y, z, ow, oh, od))
                placements[orig_idx] = (0, x, y, z, ow, oh, od)
            else:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any None placements (due to timeout)
        for i in range(len(boxes)):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    # First pass: BFD with original order
    best_placements = try_pack(indexed_boxes)
    
    # Calculate score
    def calc_score(placements):
        total_vol = 0
        for bin_id, x, y, z, w, h, d in placements:
            if bin_id == 0:
                total_vol += w * h * d
        return total_vol / (W * H * D)
    
    best_score = calc_score(best_placements)
    
    # Try a few more permutations if time allows
    # Use a few random shuffles of the sorted list to explore different orderings
    num_tries = 5
    rng = random.Random(42)
    
    for t in range(num_tries):
        if time.time() > deadline - 15:
            break
        
        # Create a modified order: keep top 50% in order, shuffle the rest
        n = len(indexed_boxes)
        mid = n // 2
        modified = indexed_boxes[:mid] + indexed_boxes[mid:]
        rng.shuffle(modified[mid:])
        
        placements = try_pack(modified)
        score = calc_score(placements)
        
        if score > best_score:
            best_score = score
            best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
