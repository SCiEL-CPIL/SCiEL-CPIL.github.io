"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Generate all 6 rotations for each box
    def get_rotations(w, h, d):
        rots = set()
        for perm in permutations((w, h, d)):
            rots.add(perm)
        return list(rots)

    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]

    # Sort boxes by volume descending for better packing
    box_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    def try_pack(order, seed):
        """Try to pack boxes in given order using skyline approach."""
        rng = random.Random(seed)
        
        # Skyline: list of (x, y, height) representing the top surface
        # We'll use a 2D grid approach for the skyline
        # Represent the skyline as a list of segments along x for each "row" in y
        # Actually, let's use a simpler approach: maintain a list of placed boxes
        # and for each new box, find the best (x, y, z) position
        
        placed = []  # list of (x, y, z, w, h, d)
        result = [None] * n
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = -1
            
            # Try all rotations
            for rw, rh, rd in box_rots[idx]:
                # Find best position using skyline-like approach
                # Try placing at various x, y positions
                # Use a grid-based search with step size based on box size
                
                # Determine search granularity
                # Try corners and offsets from existing boxes
                
                candidate_positions = set()
                
                # Always try origin
                candidate_positions.add((0, 0))
                
                # Try positions based on existing placed boxes
                for (px, py, pz, pw, ph, pd) in placed:
                    # Right of existing box
                    candidate_positions.add((px + pw, py))
                    # Above existing box (in y)
                    candidate_positions.add((px, py + ph))
                    
                # Also try some random positions
                for _ in range(5):
                    rx = rng.randint(0, max(0, W - rw))
                    ry = rng.randint(0, max(0, H - rh))
                    candidate_positions.add((rx, ry))
                
                for (cx, cy) in candidate_positions:
                    # Check if box fits in x, y
                    if cx + rw > W or cy + rh > H:
                        continue
                    
                    # Find the lowest z where the box fits
                    # Check overlap with all placed boxes
                    min_z = 0
                    valid = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        # Check x-y overlap
                        if cx < px + pw and cx + rw > px and cy < py + ph and cy + rh > py:
                            # There is overlap in x-y, so z must be above this box
                            min_z = max(min_z, pz + pd)
                    
                    if min_z + rd > D:
                        continue
                    
                    # Valid position found
                    # Score: prefer lower z, then lower y, then lower x
                    score = -(min_z * 1000 + cy * 10 + cx)
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, min_z, rw, rh, rd)
            
            if best_pos is not None:
                x, y, z, rw, rh, rd = best_pos
                placed.append((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        return result

    def calculate_score(placements):
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        return vol / container_vol

    best_placements = None
    best_score = -1

    # First pass: sorted order
    placements = try_pack(box_order, 42)
    score = calculate_score(placements)
    if score > best_score:
        best_score = score
        best_placements = placements

    # Try different random seeds and orderings
    num_attempts = 20
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        seed = 42 + attempt
        rng = random.Random(seed)
        
        # Create a perturbed order
        order = list(box_order)
        # Partially shuffle the smaller boxes
        if len(order) > 10:
            # Keep top 10 in order, shuffle the rest
            top = order[:10]
            rest = order[10:]
            rng.shuffle(rest)
            order = top + rest
        
        placements = try_pack(order, seed)
        score = calculate_score(placements)
        if score > best_score:
            best_score = score
            best_placements = placements

    # Local improvement: try to move some boxes
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 5:
                    break
                if best_placements[idx][0] == -1:
                    continue
                x, y, z, rw, rh, rd = best_placements[idx][1:]
                # Try to move this box to a better position
                # Remove it temporarily
                other_placed = [p for i, p in enumerate(best_placements) if i != idx and p[0] == 0]
                
                best_new_pos = None
                best_new_score = -1
                for rw2, rh2, rd2 in box_rots[idx]:
                    candidate_positions = set()
                    candidate_positions.add((0, 0))
                    for (px, py, pz, pw, ph, pd) in other_placed:
                        candidate_positions.add((px + pw, py))
                        candidate_positions.add((px, py + ph))
                    
                    for (cx, cy) in candidate_positions:
                        if cx + rw2 > W or cy + rh2 > H:
                            continue
                        min_z = 0
                        valid = True
                        for (px, py, pz, pw, ph, pd) in other_placed:
                            if cx < px + pw and cx + rw2 > px and cy < py + ph and cy + rh2 > py:
                                min_z = max(min_z, pz + pd)
                        if min_z + rd2 > D:
                            continue
                        score = -(min_z * 1000 + cy * 10 + cx)
                        if score > best_new_score:
                            best_new_score = score
                            best_new_pos = (cx, cy, min_z, rw2, rh2, rd2)
                
                if best_new_pos is not None:
                    new_x, new_y, new_z, new_rw, new_rh, new_rd = best_new_pos
                    old_x, old_y, old_z, old_rw, old_rh, old_rd = x, y, z, rw, rh, rd
                    # Only update if it improves (lower z)
                    if new_z < old_z or (new_z == old_z and (new_y < old_y or (new_y == old_y and new_x < old_x))):
                        best_placements[idx] = (0, new_x, new_y, new_z, new_rw, new_rh, new_rd)
                        improved = True

    return best_placements
# EVOLVE-BLOCK-END
