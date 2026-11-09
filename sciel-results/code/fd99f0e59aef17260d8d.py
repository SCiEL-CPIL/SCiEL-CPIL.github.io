import time
import random
from collections import defaultdict

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # 1. Prepare boxes with indices and sort by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placements = [None] * n
    placed_boxes = []  # List of (x, y, z, w, h, d)
    
    # Grid acceleration for overlap checking
    # Use a 3D grid to quickly find potentially overlapping boxes
    # Cell size: use the minimum dimension of any box to ensure accuracy
    min_dim = min(min(w, h, d) for w, h, d in boxes) if boxes else 1
    cell_size = max(1, min_dim // 2)
    
    # Grid: maps (gx, gy, gz) -> list of box indices
    grid = defaultdict(list)
    
    def get_rotations(w, h, d):
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)
    
    def get_grid_cells(x, y, z, bw, bh, bd):
        """Get all grid cells that a box occupies."""
        gx1 = x // cell_size
        gy1 = y // cell_size
        gz1 = z // cell_size
        gx2 = (x + bw - 1) // cell_size
        gy2 = (y + bh - 1) // cell_size
        gz2 = (z + bd - 1) // cell_size
        cells = set()
        for gx in range(gx1, gx2 + 1):
            for gy in range(gy1, gy2 + 1):
                for gz in range(gz1, gz2 + 1):
                    cells.add((gx, gy, gz))
        return cells
    
    def fits_fast(x, y, z, bw, bh, bd):
        """Fast overlap check using grid."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        
        # Get all grid cells the new box would occupy
        new_cells = get_grid_cells(x, y, z, bw, bh, bd)
        
        # Check each cell for existing boxes
        checked = set()
        for cell in new_cells:
            for box_idx in grid[cell]:
                if box_idx in checked:
                    continue
                checked.add(box_idx)
                px, py, pz, pw, ph, pd = placed_boxes[box_idx]
                # Check if boxes overlap
                if not (x + bw <= px or px + pw <= x or 
                        y + bh <= py or py + ph <= y or 
                        z + bd <= pz or pz + pd <= z):
                    return False
        return True
    
    def add_to_grid(box_idx, x, y, z, bw, bh, bd):
        """Add a box to the grid."""
        cells = get_grid_cells(x, y, z, bw, bh, bd)
        for cell in cells:
            grid[cell].append(box_idx)
    
    def remove_from_grid(box_idx, x, y, z, bw, bh, bd):
        """Remove a box from the grid."""
        cells = get_grid_cells(x, y, z, bw, bh, bd)
        for cell in cells:
            if box_idx in grid[cell]:
                grid[cell].remove(box_idx)
    
    # Candidate points: set for uniqueness, list for iteration
    candidate_set = {(0, 0, 0)}
    
    def score_position(x, y, z, bw, bh, bd):
        """Score a position - lower is better (bottom-left-front)."""
        # Primary: minimize z, then y, then x
        # Secondary: prefer positions that fill more volume near origin
        return (z, y, x)
    
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break

        rotations = get_rotations(w, h, d)
        
        # Sort candidates by score
        sorted_candidates = sorted(candidate_set, key=lambda p: score_position(p[0], p[1], p[2], 0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_score = None
        
        found = False
        for cx, cy, cz in sorted_candidates:
            if time.time() > deadline:
                break
                
            for rw, rh, rd in rotations:
                if fits_fast(cx, cy, cz, rw, rh, rd):
                    pos_score = score_position(cx, cy, cz, rw, rh, rd)
                    if best_score is None or pos_score < best_score:
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
                        best_score = pos_score
                    # For bottom-left-front, first valid position is best
                    # But let's check a few more to be safe
                    found = True
                    break
            if found:
                break
        
        if found:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            
            # Place the box
            placed_boxes.append((bx, by, bz, bw, bh, bd))
            placements[i] = (0, bx, by, bz, bw, bh, bd)
            add_to_grid(len(placed_boxes) - 1, bx, by, bz, bw, bh, bd)
            
            # Update candidates
            candidate_set.discard((bx, by, bz))
            
            # Add new corner candidates
            new_cands = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd)
            ]
            
            for nc in new_cands:
                candidate_set.add(nc)
            
        else:
            placements[i] = (-1, 0, 0, 0, w, h, d)

    # Fill any remaining None (if loop broke early due to time)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
