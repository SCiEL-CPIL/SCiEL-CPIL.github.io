"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Precompute all 6 rotations for each box
    # Rotation is a permutation of (w, h, d)
    def get_rotations(w, h, d):
        return [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
    
    # Sort boxes by volume descending, then by max dimension
    box_volumes = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: (-box_volumes[i], -max(boxes[i])))
    
    # For each box, precompute best rotation (minimize longest side first? or just keep all)
    # We'll try multiple strategies. Let's use a simple shelf packing with rotation choice.
    # Strategy: For each box in sorted order, try all 6 rotations and pick the one that fits best.
    # "Fits best" heuristic: minimize the "wasted" space in the current shelf/layer.
    
    # Simple 3D shelf algorithm:
    # - Process boxes in order (sorted by volume desc)
    # - Maintain current position (x, y, z), current row height, current layer depth
    # - For each box, try all rotations
    # - For each rotation, check if it fits in current position
    #   - If yes, place it
    #   - If not, try to move to next position in same layer, next layer, etc.
    # - If no rotation fits anywhere, mark as left out
    
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # State: list of placed boxes with their positions
    placed = []  # (box_idx, x, y, z, w, h, d)
    
    def can_place(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with all placed boxes
        for (bi, px, py, pz, pw, ph, pd) in placed:
            # Check if boxes overlap
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return False
        return True
    
    def try_place(box_idx):
        w0, h0, d0 = boxes[box_idx]
        rots = get_rotations(w0, h0, d0)
        
        # Try to find a position for each rotation
        # Heuristic: try to pack from bottom-left-front, row by row, layer by layer
        # We'll use a simple approach: try positions in a grid-like fashion
        
        best_pos = None
        best_score = -1
        
        # Generate candidate positions
        # For simplicity, we'll try:
        # 1. Current "cursor" position
        # 2. Positions along the x-axis in the current row
        # 3. Positions in new rows/layers
        
        # Let's use a simpler heuristic: try to place at the earliest available position
        # Scan through possible positions
        
        # To keep it efficient, we'll limit the search
        # Try positions in order of increasing (z, y, x)
        
        # First, get all placed boxes to determine occupied regions
        # For each rotation, try to find the "lowest" position
        
        for rot in rots:
            rw, rh, rd = rot
            
            # Try to find a position
            # Start from z=0, y=0, x=0 and increment
            # This is O(W*H*D) which is too slow for large containers
            
            # Better: use a greedy approach
            # Try x=0, y=0, z=0
            # If it doesn't fit, try x+1, etc.
            
            # For efficiency, we'll use a limited search
            # Try positions where x, y, z are at boundaries of existing boxes
            
            candidate_positions = [(0, 0, 0)]
            
            # Add positions based on existing boxes
            for (bi, px, py, pz, pw, ph, pd) in placed:
                candidate_positions.append((px + pw, py, pz))
                candidate_positions.append((px, py + ph, pz))
                candidate_positions.append((px, py, pz + pd))
            
            # Also try some offset positions
            for (cx, cy, cz) in candidate_positions[:50]:  # Limit to 50 candidates
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                if can_place(cx, cy, cz, rw, rh, rd):
                    # Score: prefer lower z, then lower y, then lower x
                    score = -(cz * W * H + cy * W + cx)
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, rw, rh, rd)
        
        if best_pos:
            cx, cy, cz, rw, rh, rd = best_pos
            placed.append((box_idx, cx, cy, cz, rw, rh, rd))
            placements[box_idx] = (0, cx, cy, cz, rw, rh, rd)
            return True
        return False
    
    # Process boxes in sorted order
    for box_idx in sorted_indices:
        try_place(box_idx)
    
    return placements
# EVOLVE-BLOCK-END
