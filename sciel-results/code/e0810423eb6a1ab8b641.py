"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def pack(boxes, container):
    """
    3D Bin Packing: place boxes (no rotation) into numbered bins.
    Objective: minimize mean B + 0.1*(1 - last_bin_occupied_volume/container_volume)
    where B = number of occupied bins.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    W, H, D = container
    container_volume = W * H * D
    
    # Sort boxes by volume descending (Best-Fit Decreasing style)
    # We need to track original indices to return in correct order
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # We'll use a shelf-based approach with multiple strategies and pick the best
    # Strategy: for each box, try to fit it in existing bins first (best fit),
    # then open new bins. Use shelf packing within each bin.
    
    def try_shelf_pack(sorted_boxes, max_time=110):
        """Shelf packing with best-fit bin selection."""
        bins = []  # each bin is a list of (x, y, z, w, h, d)
        bin_shelves = []  # each bin's shelves: list of (y, height, next_x)
        
        for idx, (orig_idx, w, h, d, vol) in enumerate(sorted_boxes):
            best_bin = -1
            best_score = float('inf')
            best_pos = None
            
            # Try to fit in existing bins
            for b_idx in range(len(bins)):
                shelves = bin_shelves[b_idx]
                # Try each shelf
                for s_idx, (shelf_y, shelf_h, next_x) in enumerate(shelves):
                    if shelf_y + h <= H and next_x + w <= W:
                        # Check if this position is actually free (no overlap with other boxes)
                        # In shelf packing, we assume the shelf is clear up to next_x
                        # Position: (next_x, shelf_y, 0) - but z needs to be handled
                        # Actually in shelf packing, z is determined by the shelf layer
                        # Let me reconsider: shelf packing works in 2D (x-y) per layer (z)
                        
                        # Simplified: treat as 2D shelf in x-y plane, stack layers in z
                        # For now, place at (next_x, shelf_y, 0) and check z separately
                        # This is getting complex. Let me use a simpler approach.
                        pass
                
                # Try opening a new shelf in this bin
                # Find the top of the current bin
                if shelves:
                    top_y = max(s[0] + s[1] for s in shelves)
                else:
                    top_y = 0
                
                if top_y + h <= H:
                    # New shelf at top_y
                    if 0 + w <= W:  # always true if w <= W
                        score = top_y  # prefer lower shelves
                        if score < best_score:
                            best_score = score
                            best_bin = b_idx
                            best_pos = (0, top_y, 0, w, h, d)
                            # Mark this as a new shelf
                            best_pos = ('new_shelf', top_y)
            
            # If no existing bin works, or we decide to open a new bin
            if best_bin == -1:
                # Open a new bin
                bins.append([])
                bin_shelves.append([])
                best_bin = len(bins) - 1
                best_pos = ('new_shelf', 0)
            
            if isinstance(best_pos, tuple) and best_pos[0] == 'new_shelf':
                new_y = best_pos[1]
                bin_shelves[best_bin].append((new_y, h, w))  # (y, height, next_x)
                bins[best_bin].append((0, new_y, 0, w, h, d))
            else:
                # This case shouldn't happen with current logic
                pass
        
        return bins, bin_shelves
    
    # Let me implement a cleaner approach: First Fit Decreasing with shelf packing
    # Each bin maintains shelves. A shelf is a horizontal slice at some z-level.
    # Within a shelf, boxes are placed side by side along x.
    # When a shelf is full in x or a box doesn't fit, start a new shelf above.
    
    def first_fit_decreasing_shelf(sorted_boxes):
        """
        FFD with shelf packing.
        Each bin has a list of shelves. Each shelf has:
        - z: the z-coordinate of the bottom of the shelf
        - height: the height of the shelf (max height of boxes in it)
        - boxes: list of (x, w) for boxes in this shelf
        - next_x: next available x position
        """
        bins = []  # list of bins, each bin is list of shelves
        # Each shelf: {'z': int, 'height': int, 'next_x': int, 'boxes': [(x, w, h, d)]}
        
        for orig_idx, w, h, d, vol in sorted_boxes:
            placed = False
            best_bin_idx = -1
            best_shelf_idx = -1
            best_fit_score = float('inf')
            
            for b_idx, shelves in enumerate(bins):
                # Try each existing shelf
                for s_idx, shelf in enumerate(shelves):
                    if shelf['next_x'] + w <= W:
                        # Box fits in this shelf
                        # Score: prefer shelves that are more "full" or lower
                        # Use the gap as a score (smaller gap = better fit)
                        gap = W - (shelf['next_x'] + w)
                        # Also consider z-level (lower is better for filling)
                        score = gap * 1000 + shelf['z']
                        if score < best_fit_score:
                            best_fit_score = score
                            best_bin_idx = b_idx
                            best_shelf_idx = s_idx
                
                # Try opening a new shelf in this bin
                if shelves:
                    top_z = max(s['z'] + s['height'] for s in shelves)
                else:
                    top_z = 0
                
                if top_z + h <= D:
                    # New shelf at top_z
                    if w <= W:
                        score = top_z * 1000  # prefer lower new shelves
                        if score < best_fit_score:
                            best_fit_score = score
                            best_bin_idx = b_idx
                            best_shelf_idx = -1  # signal new shelf
                            # Store the new shelf z
                            new_shelf_z = top_z
            
            if best_bin_idx == -1:
                # Open a new bin
                bins.append([])
                best_bin_idx = len(bins) - 1
                best_shelf_idx = -1
                new_shelf_z = 0
            
            if best_shelf_idx == -1:
                # New shelf
                new_shelf = {
                    'z': new_shelf_z,
                    'height': h,
                    'next_x': w,
                    'boxes': [(0, w, h, d)]
                }
                bins[best_bin_idx].append(new_shelf)
            else:
                shelf = bins[best_bin_idx][best_shelf_idx]
                x = shelf['next_x']
                shelf['boxes'].append((x, w, h, d))
                shelf['next_x'] += w
                shelf['height'] = max(shelf['height'], h)
        
        return bins
    
    def compute_score(bins, W, H, D):
        """Compute the score: mean B + 0.1*(1 - last_bin_vol/container_vol)"""
        if not bins:
            return 0.0
        B = len(bins)
        # Last bin is the one with the highest index
        last_bin = bins[-1]
        last_bin_vol = 0
        for shelf in last_bin:
            for (x, w, h, d) in shelf['boxes']:
                last_bin_vol += w * h * d
        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        occupied_frac = last_bin_vol / container_vol
        score = B + 0.1 * (1 - occupied_frac)
        return score
    
    # Run FFD shelf packing
    bins = first_fit_decreasing_shelf(sorted_boxes)
    
    # Convert to placement format
    placements = [None] * len(boxes)
    for b_idx, shelves in enumerate(bins):
        for shelf in shelves:
            for (x, w, h, d) in shelf['boxes']:
                z = shelf['z']
                # Find the original index
                # We need to track which box this is
                # Let me restructure to track original indices
    
    # Let me redo this with proper tracking
    def first_fit_decreasing_shelf_tracked(sorted_boxes):
        bins = []
        placements = [None] * len(sorted_boxes)  # indexed by position in sorted order
        
        for pos, (orig_idx, w, h, d, vol) in enumerate(sorted_boxes):
            best_bin_idx = -1
            best_shelf_idx = -1
            best_fit_score = float('inf')
            new_shelf_z = 0
            
            for b_idx, shelves in enumerate(bins):
                for s_idx, shelf in enumerate(shelves):
                    if shelf['next_x'] + w <= W:
                        gap = W - (shelf['next_x'] + w)
                        score = gap * 1000 + shelf['z']
                        if score < best_fit_score:
                            best_fit_score = score
                            best_bin_idx = b_idx
                            best_shelf_idx = s_idx
                
                if shelves:
                    top_z = max(s['z'] + s['height'] for s in shelves)
                else:
                    top_z = 0
                
                if top_z + h <= D and w <= W:
                    score = top_z * 1000
                    if score < best_fit_score:
                        best_fit_score = score
                        best_bin_idx = b_idx
                        best_shelf_idx = -1
                        new_shelf_z = top_z
            
            if best_bin_idx == -1:
                bins.append([])
                best_bin_idx = len(bins) - 1
                best_shelf_idx = -1
                new_shelf_z = 0
            
            if best_shelf_idx == -1:
                new_shelf = {
                    'z': new_shelf_z,
                    'height': h,
                    'next_x': w,
                    'boxes': [(0, w, h, d, orig_idx)]
                }
                bins[best_bin_idx].append(new_shelf)
                placements[pos] = (best_bin_idx, 0, new_shelf_z, 0, w, h, d)
            else:
                shelf = bins[best_bin_idx][best_shelf_idx]
                x = shelf['next_x']
                shelf['boxes'].append((x, w, h, d, orig_idx))
                shelf['next_x'] += w
                shelf['height'] = max(shelf['height'], h)
                placements[pos] = (best_bin_idx, x, new_shelf_z, 0, w, h, d)
        
        return bins, placements
    
    bins, sorted_placements = first_fit_decreasing_shelf_tracked(sorted_boxes)
    
    # Map back to original order
    final_placements = [None] * len(boxes)
    for pos, (orig_idx, w, h, d, vol) in enumerate(sorted_boxes):
        final_placements[orig_idx] = sorted_placements[pos]
    
    return final_placements
# EVOLVE-BLOCK-END
