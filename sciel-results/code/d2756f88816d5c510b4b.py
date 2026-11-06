"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def load(boxes, container):
    W, H, D = int(container[0]), int(container[1]), int(container[2])
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Deadline guard: 115s to be safe under 120s limit
    deadline = time.time() + 115.0
    
    # Pre-compute unique orientations for each box
    # For each box (w, h, d), generate all distinct permutations
    box_orients = []
    for i, (w, h, d) in enumerate(boxes):
        dims = sorted([w, h, d])
        seen = set()
        orients = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        box_orients.append(orients)
    
    # Heuristic ordering: sort by volume descending, then by max dimension
    # This tends to place large boxes first which is important for packing
    def get_order(boxes, box_orients, rng=None):
        indices = list(range(len(boxes)))
        # Sort by volume descending, then by largest dimension descending
        indices.sort(key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                     -max(boxes[i])), reverse=False)
        # Actually sort properly: largest volume first
        indices.sort(key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]), reverse=True)
        return indices
    
    def pack_with_order(order, boxes, box_orients, W, H, D):
        """
        Greedy placement using a list of placed boxes.
        For each box in order, try all orientations and all candidate positions.
        Candidate positions are generated from the "bottom-left-front" corner
        and corners of already-placed boxes.
        """
        placed = []  # list of (x, y, z, w, h, d)
        result = [-1] * len(boxes)  # default: not placed
        
        for idx in order:
            w_orig, h_orig, d_orig = boxes[idx]
            best_pos = None
            best_vol = -1
            best_orient = None
            
            # Generate candidate positions
            # Start with origin
            candidates = [(0, 0, 0)]
            
            # Add positions derived from placed boxes
            for (px, py, pz, pw, ph, pd) in placed:
                # To the right of placed box
                candidates.append((px + pw, py, pz))
                # Above placed box
                candidates.append((px, py + ph, pz))
                # In front of placed box
                candidates.append((px, py, pz + pd))
                # Combinations
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))
            
            # Deduplicate candidates
            candidates = list(set(candidates))
            
            # For each orientation, try each candidate position
            for ow, oh, od in box_orients[idx]:
                for (cx, cy, cz) in candidates:
                    # Check bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    # Check overlap with all placed boxes
                    valid = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        if (cx < px + pw and cx + ow > px and
                            cy < py + ph and cy + oh > py and
                            cz < pz + pd and cz + od > pz):
                            valid = False
                            break
                    if valid:
                        # Valid placement - prefer lower z, then lower y, then lower x
                        # to keep things compact
                        score = -(cz * 1000000 + cy * 1000 + cx)
                        if score > best_vol:
                            best_vol = score
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                placed.append((cx, cy, cz, ow, oh, od))
                result[idx] = (0, cx, cy, cz, ow, oh, od)
            # else: box not placed, result[idx] stays -1
        
        return result
    
    def compute_score(placement, boxes, W, H, D):
        """Compute fraction of container volume filled."""
        total_vol = W * H * D
        if total_vol == 0:
            return 0.0
        filled = 0
        for p in placement:
            if p[0] == 0:
                _, x, y, z, w, h, d = p
                filled += w * h * d
        return filled / total_vol
    
    # Strategy: Try multiple orderings and pick the best
    best_placement = None
    best_score = -1.0
    
    # Initial ordering: by volume descending
    order1 = get_order(boxes, box_orients)
    placement1 = pack_with_order(order1, boxes, box_orients, W, H, D)
    score1 = compute_score(placement1, boxes, W, H, D)
    best_placement = placement1
    best_score = score1
    
    # Try a few more orderings with time budget
    rng = random.Random(42)
    
    # Ordering 2: by max dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: max(boxes[i]), reverse=True)
    if time.time() < deadline:
        placement2 = pack_with_order(order2, boxes, box_orients, W, H, D)
        score2 = compute_score(placement2, boxes, W, H, D)
        if score2 > best_score:
            best_score = score2
            best_placement = placement2
    
    # Ordering 3: by min dimension descending (taller boxes first)
    order3 = list(range(n))
    order3.sort(key=lambda i: min(boxes[i]), reverse=True)
    if time.time() < deadline:
        placement3 = pack_with_order(order3, boxes, box_orients, W, H, D)
        score3 = compute_score(placement3, boxes, W, H, D)
        if score3 > best_score:
            best_score = score3
            best_placement = placement3
    
    # Ordering 4: random shuffles of the volume-sorted order
    for attempt in range(20):
        if time.time() >= deadline - 5:
            break
        order_rand = list(order1)
        rng.shuffle(order_rand)
        placement_r = pack_with_order(order_rand, boxes, box_orients, W, H, D)
        score_r = compute_score(placement_r, boxes, W, H, D)
        if score_r > best_score:
            best_score = score_r
            best_placement = placement_r
    
    # ILS: perturb best ordering and re-pack
    if best_placement is not None and time.time() < deadline - 10:
        # Reconstruct order from best placement
        # Actually, we need to track which order produced best_placement
        # Let's just do a few perturbations on order1
        current_order = list(order1)
        current_score = best_score
        
        for iteration in range(50):
            if time.time() >= deadline - 5:
                break
            
            # Perturbation: swap two random elements
            new_order = list(current_order)
            i, j = rng.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            placement_new = pack_with_order(new_order, boxes, box_orients, W, H, D)
            score_new = compute_score(placement_new, boxes, W, H, D)
            
            if score_new >= current_score:
                current_order = new_order
                current_score = score_new
                if score_new > best_score:
                    best_score = score_new
                    best_placement = placement_new
    
    # Final safety: ensure we return a valid placement
    if best_placement is None:
        best_placement = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placement
# EVOLVE-BLOCK-END
