import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n_boxes = len(boxes)
    container_vol = W * H * D
    
    # Column-based 2D packing: track max Z-height per (x,y) column
    # This decouples vertical stacking from horizontal placement
    
    def column_pack(ordered_indices):
        """
        2D column-based packing:
        - Maintain a 2D grid of max Z-heights per (x,y) column
        - For each box, find the bottom-left-most (x,y) position where
          the box fits (all columns in footprint have enough remaining height)
        - Place box at the lowest feasible Z (min of column heights in footprint)
        - Update column heights
        """
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        
        # Column height grid: col_height[x, y] = max Z occupied at column (x,y)
        col_height = np.zeros((W, H), dtype=np.int32)
        
        # Candidate positions: sorted by (z, y, x) for bottom-left-back priority
        # We'll generate candidates from placed boxes
        candidates = [(0, 0)]  # (x, y) positions to try
        
        for idx in ordered_indices:
            if time.time() > deadline:
                return results, False
            
            w, h, d = boxes[idx]
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            rotations = list(set(itertools.permutations((w, h, d))))
            
            for rw, rh, rd in rotations:
                # Try each candidate (x, y) position
                for cx, cy in candidates:
                    if cx + rw > W or cy + rh > H:
                        continue
                    
                    # Check if box fits in this column region
                    # Get the max height in the footprint region
                    region = col_height[cx:cx+rw, cy:cy+rh]
                    max_h_in_region = int(np.max(region)) if region.size > 0 else 0
                    
                    # The box can be placed at z = max_h_in_region
                    # It fits if max_h_in_region + rd <= D
                    cz = max_h_in_region
                    if cz + rd > D:
                        continue
                    
                    # This is a valid placement
                    # Score: prioritize lower z, then smaller y, then smaller x
                    score = cz * 1000000 + cy * 1000 + cx
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                
                # Update column heights
                col_height[bx:bx+bw, by:by+bh] = np.maximum(
                    col_height[bx:bx+bw, by:by+bh], bz + bd
                )
                
                # Generate new candidate positions
                # Right edge
                candidates.append((bx + bw, by))
                # Front edge
                candidates.append((bx, by + bh))
                # Corner
                candidates.append((bx + bw, by + bh))
                # Also try positions just above existing boxes
                # For each column in the footprint, the new height is bz+bd
                # New candidates at (x, y) where a box ends at that column
                for dx in range(bw):
                    for dy in range(bh):
                        nx = bx + dx
                        ny = by + dy
                        # Add (nx+1, ny) and (nx, ny+1) as potential new positions
                        if nx + 1 < W:
                            candidates.append((nx + 1, ny))
                        if ny + 1 < H:
                            candidates.append((nx, ny + 1))
                
                # Deduplicate candidates periodically to save memory/time
                if len(candidates) > 5000:
                    candidates = list(set(candidates))
        
        return results, True

    def calculate_score(results):
        total_vol = 0
        for bin_id, x, y, z, w, h, d in results:
            if bin_id == 0:
                total_vol += w * h * d
        return total_vol

    best_results = None
    best_score = 0
    
    # Define sorting strategies
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i]),                              # Max Dimension
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2],  # Surface Area
        lambda i: boxes[i][0] * boxes[i][1],                  # Base Area (footprint)
        lambda i: -min(boxes[i]),                             # Min Dimension
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2] / max(1, min(boxes[i])),  # Volume / min dim
        lambda i: max(boxes[i]) ** 2,                          # Max dim squared
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],     # Sum of dims
    ]
    
    indices = list(range(n_boxes))
    
    # Try each strategy
    for sort_func in strategies:
        if time.time() > deadline:
            break
            
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        
        results, completed = column_pack(sorted_indices)
        
        if completed:
            score = calculate_score(results)
            if score > best_score:
                best_score = score
                best_results = results
    
    # If we have time left, try random restarts
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 30:
            random.shuffle(indices)
            
            results, completed = column_pack(indices)
            attempts += 1
            
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results

    # Fallback if no boxes were placed
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
