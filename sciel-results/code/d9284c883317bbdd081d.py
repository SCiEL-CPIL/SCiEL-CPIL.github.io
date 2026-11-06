"""Improved 3D Container Loading using Best-Fit Decreasing with rotations and time guard."""
import time
import itertools

# EVOLVE-BLOCK-START

def _all_rotations(w, h, d):
    """Generate all unique rotations (permutations) of box dimensions."""
    perms = set(itertools.permutations((w, h, d)))
    return list(perms)

def _fits(box, pos, placed, W, H, D):
    """Check if box at pos overlaps with any placed box and is within container."""
    bw, bh, bd = box
    x, y, z = pos
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Check overlap in 3D
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-(x[1] * x[2] * x[3]), -max(x[1], x[2], x[3])))
    
    # Precompute rotations for each box
    box_rotations = {}
    for i, w, h, d in indexed_boxes:
        box_rotations[i] = _all_rotations(w, h, d)
    
    # Greedy Best-Fit Decreasing with rotations
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * n
    
    for idx, (orig_idx, w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_pos = None
        best_score = None
        best_rot = None
        
        # Generate candidate positions:
        # 1. Origin
        # 2. On top of each placed box
        # 3. To the right of each placed box
        # 4. In front of each placed box
        
        candidate_positions = set()
        candidate_positions.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            candidate_positions.add((px + pw, py, pz))
            candidate_positions.add((px, py + ph, pz))
            candidate_positions.add((px, py, pz + pd))
        
        # Limit candidate positions for performance
        if len(candidate_positions) > 500:
            # Keep only a subset - prioritize lower z, then y, then x
            candidate_positions = set(sorted(candidate_positions, key=lambda p: (p[2], p[1], p[0]))[:500])
        
        for rot in box_rotations[orig_idx]:
            rw, rh, rd = rot
            for (cx, cy, cz) in candidate_positions:
                if not _fits(rot, (cx, cy, cz), placed, W, H, D):
                    continue
                # Score: prefer lower z, then lower y, then lower x (best fit)
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = rot
        
        if best_pos is not None:
            placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
            result[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
        else:
            result[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill any remaining None entries (shouldn't happen, but safety)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result

# EVOLVE-BLOCK-END
