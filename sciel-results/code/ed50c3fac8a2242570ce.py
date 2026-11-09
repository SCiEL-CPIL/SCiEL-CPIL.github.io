"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = start_time + 145  # 145s deadline, 5s safety margin
    W, H, D = container
    n = len(boxes)
    
    # Precompute all 6 rotations for each box (unique dimensions)
    def get_rotations(w, h, d):
        rots = set()
        for perm in permutations((w, h, d)):
            rots.add(perm)
        return list(rots)
    
    rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    volumes = [w * h * d for w, h, d in boxes]
    
    # Sort boxes by volume descending for better packing heuristics
    order = sorted(range(n), key=lambda i: -volumes[i])
    
    def can_place(x, y, z, bw, bh, bd, placed):
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True
    
    def generate_positions(placed):
        """Generate candidate positions for the next box."""
        positions = set()
        # Origin
        positions.add((0, 0, 0))
        
        # For each placed box, generate positions adjacent to it
        for px, py, pz, pw, ph, pd in placed:
            # Positions touching the placed box on each face
            candidates = [
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
                # Also try positions at the corners for better packing
                (px + pw, py + ph, pz),
                (px + pw, py, pz + pd),
                (px, py + ph, pz + pd),
                (px + pw, py + ph, pz + pd),
            ]
            for cx, cy, cz in candidates:
                if cx < W and cy < H and cz < D:
                    positions.add((cx, cy, cz))
        
        # Sort positions by a heuristic: prefer positions closer to origin
        # (x + y + z as a proxy for "lower" in the container)
        pos_list = list(positions)
        pos_list.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
        return pos_list
    
    # Best solution tracking
    best_loaded_count = 0
    best_placement = None
    
    # Greedy with rotation search
    placed = []
    loaded = [False] * n
    loaded_count = 0
    
    for idx in order:
        if time.time() > DEADLINE:
            break
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        
        for rot in rotations[idx]:
            bw, bh, bd = rot
            if bw > W or bh > H or bd > D:
                continue
            
            positions = generate_positions(placed)
            for (x, y, z) in positions:
                if can_place(x, y, z, bw, bh, bd, placed):
                    best_pos = (x, y, z)
                    best_rot = rot
                    break
            if best_pos is not None:
                break
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placed.append((x, y, z, bw, bh, bd))
            loaded[idx] = True
            loaded_count += 1
    
    # Build result
    placements = []
    placed_idx = 0
    for i in range(n):
        if loaded[i]:
            # Find the placement for this box
            # We need to map back - let's redo this more carefully
            pass
    
    # Actually, let's track which box went where
    # Redo with proper tracking
    placed = []  # list of (box_idx, x, y, z, bw, bh, bd)
    loaded = [False] * n
    loaded_count = 0
    
    for idx in order:
        if time.time() > DEADLINE:
            break
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        
        for rot in rotations[idx]:
            bw, bh, bd = rot
            if bw > W or bh > H or bd > D:
                continue
            
            positions = generate_positions([(x, y, z, pw, ph, pd) for (_, x, y, z, pw, ph, pd) in placed])
            for (x, y, z) in positions:
                if can_place(x, y, z, bw, bh, bd, [(x2, y2, z2, pw2, ph2, pd2) for (_, x2, y2, z2, pw2, ph2, pd2) in placed]):
                    best_pos = (x, y, z)
                    best_rot = rot
                    break
            if best_pos is not None:
                break
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placed.append((idx, x, y, z, bw, bh, bd))
            loaded[idx] = True
            loaded_count += 1
    
    # Build result
    placements = []
    placed_by_idx = {p[0]: p for p in placed}
    for i in range(n):
        if i in placed_by_idx:
            _, x, y, z, bw, bh, bd = placed_by_idx[i]
            placements.append((0, x, y, z, bw, bh, bd))
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
