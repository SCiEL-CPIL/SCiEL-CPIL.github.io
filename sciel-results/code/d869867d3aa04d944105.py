"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Voxel grid resolution: use 1 unit voxels for exact geometry
    # Each bin has its own occupancy grid
    bins = []          # list of lists of (x, y, z, w, h, d)
    bin_grids = []     # list of 3D numpy boolean arrays
    bin_bboxes = []    # (max_x, max_y, max_z) per bin
    
    def make_grid():
        return np.zeros((W, H, D), dtype=bool)
    
    def can_place_grid(grid, x, y, z, w, h, d):
        """O(1) collision check using voxel grid slicing."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any voxel in the target region is occupied
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        return True
    
    def place_in_grid(grid, x, y, z, w, h, d):
        """Mark voxels as occupied."""
        grid[x:x+w, y:y+h, z:z+d] = True
    
    def find_position(bin_boxes, grid, w, h, d):
        """Find the best position using candidate points from box faces."""
        candidates = set()
        candidates.add((0, 0, 0))
        
        if bin_boxes:
            for (px, py, pz, pw, ph, pd) in bin_boxes:
                # Generate candidate positions from each face
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates and score them
        valid_candidates = []
        for (x, y, z) in candidates:
            if x + w > W or y + h > H or z + d > D:
                continue
            if not can_place_grid(grid, x, y, z, w, h, d):
                continue
            # Score: prefer positions that minimize distance to origin
            # and maximize contact with existing boxes (denser packing)
            score = x + y + z
            valid_candidates.append((score, x, y, z))
        
        if not valid_candidates:
            return None
        
        valid_candidates.sort()
        return valid_candidates[0][1:]
    
    def evaluate_bin_fit(bin_boxes, grid, w, h, d):
        """Evaluate how well a box fits in a bin. Lower is better."""
        pos = find_position(bin_boxes, grid, w, h, d)
        if pos is None:
            return None, None
        x, y, z = pos
        return (x + y + z), pos
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx in range(len(bins)):
            score, pos = evaluate_bin_fit(bins[b_idx], bin_grids[b_idx], w, h, d)
            if score is not None and score < best_score:
                best_score = score
                best_bin = b_idx
                best_pos = pos
        
        if best_bin == -1:
            # Create a new bin
            bins.append([])
            bin_grids.append(make_grid())
            bin_bboxes.append((w, h, d))
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        else:
            # Update bounding box
            x, y, z = best_pos
            old_bb = bin_bboxes[best_bin]
            new_bb = (max(old_bb[0], x + w), max(old_bb[1], y + h), max(old_bb[2], z + d))
            bin_bboxes[best_bin] = new_bb
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        place_in_grid(bin_grids[best_bin], x, y, z, w, h, d)
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in new bins
            for i, (w2, h2, d2) in enumerate(boxes[idx:]):
                bin_idx = len(bins)
                bins.append([])
                bin_grids.append(make_grid())
                bin_bboxes.append((w2, h2, d2))
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                place_in_grid(bin_grids[bin_idx], 0, 0, 0, w2, h2, d2)
                placements[idx + i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Fill any None placements
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            bin_idx = len(bins)
            bins.append([])
            bin_grids.append(make_grid())
            bin_bboxes.append((w, h, d))
            bins[bin_idx].append((0, 0, 0, w, h, d))
            place_in_grid(bin_grids[bin_idx], 0, 0, 0, w, h, d)
            placements[i] = (bin_idx, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
