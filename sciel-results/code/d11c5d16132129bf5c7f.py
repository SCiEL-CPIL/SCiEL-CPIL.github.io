"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s
    
    # Precompute container volume
    container_vol = W * H * D
    
    # Sort boxes by volume descending (Best-Fit-Decreasing strategy)
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    def evaluate_solution(placements):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return 0
        # Count occupied bins
        bin_ids = set(p[0] for p in placements)
        B = len(bin_ids)
        if B == 0:
            return 0
        # Find the last bin (largest label)
        last_bin_id = max(bin_ids)
        last_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == last_bin_id)
        if container_vol == 0:
            return B
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # Skyline representation for a single bin
    # A skyline is a list of (x, y, z, w, h, d) representing maximal empty boxes.
    # Initially, the whole container is one empty box.
    
    def can_place(skyline, w, h, d):
        """Check if a box of size (w,h,d) can be placed in any slot of the skyline.
        Returns the best position (x,y,z) or None.
        Best position is defined by minimizing z, then y, then x (fill bottom, left, back).
        """
        best_pos = None
        best_cost = float('inf')
        
        for sx, sy, sz, sw, sh, sd in skyline:
            if w <= sw and h <= sh and d <= sd:
                # It fits in this slot.
                # We can place it at (sx, sy, sz).
                # Cost: prioritize lower z, then lower y, then lower x
                cost = sz * 10000 + sy * 100 + sx
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (sx, sy, sz)
        
        return best_pos

    def place_box(skyline, x, y, z, w, h, d):
        """Update the skyline after placing a box at (x,y,z) with size (w,h,d).
        Returns the new skyline.
        """
        new_skyline = []
        for sx, sy, sz, sw, sh, sd in skyline:
            # If the new box does not intersect this slot, keep it.
            # Intersection check:
            # x < sx + sw AND sx < x + w
            # y < sy + sh AND sy < y + h
            # z < sz + sd AND sz < z + d
            
            intersects = (x < sx + sw and sx < x + w and 
                          y < sy + sh and sy < y + h and 
                          z < sz + sd and sz < z + d)
            
            if not intersects:
                new_skyline.append((sx, sy, sz, sw, sh, sd))
            else:
                # The box splits this slot.
                # We need to add the parts of the slot that remain empty.
                
                # 1. Part behind the box (higher z)
                if z + d < sz + sd:
                    # Remaining depth: (sz + sd) - (z + d)
                    new_skyline.append((sx, sy, z + d, sw, sh, (sz + sd) - (z + d)))
                
                # 2. Part above the box (higher y) - only in the region where z overlaps?
                # Actually, standard skyline splitting in 3D is complex.
                # A simpler, more robust approach for small N (40 boxes) is to just use a grid or a list of placed boxes and check overlaps.
                # However, the prompt asks for a structural change. 
                # Let's stick to a simpler "List of Placed Boxes" with optimized overlap check, 
                # but use a smarter heuristic: "Best Fit" based on remaining volume in the bin.
                
                pass # We will discard the skyline approach if it's too complex to implement correctly in a short time.
        
        return new_skyline

    # Given the complexity of 3D skyline updates, let's pivot to a more robust 
    # "Best Fit Decreasing" with a global search for the best bin.
    # The current program uses a local search with random restarts. 
    # The structural change: 
    # 1. Deterministic BFD with "Best Fit" (minimizing remaining space in the bin).
    # 2. A post-processing step that tries to merge bins or move boxes to reduce B.
    
    def bin_remaining_volume(bin_boxes, W, H, D):
        """Estimate remaining volume. 
        For a rough estimate, we can use the bounding box of placed items, 
        but a better metric for 'Best Fit' is the minimum distance to the walls 
        or simply the volume of the bin minus sum of box volumes? 
        Actually, 'Best Fit' usually means fitting the box into the tightest space.
        In 3D, this is hard. 
        Let's use a simpler metric: 
        For each bin, we calculate the 'waste' or 'compactness'.
        A common heuristic is to minimize the maximum coordinate (z, then y, then x) of the placed box.
        This encourages packing towards the origin.
        """
        pass

    # Let's implement a robust BFD with a specific placement heuristic:
    # For each box, try all existing bins.
    # For each bin, find the position that minimizes (z, y, x) lexicographically.
    # This is "First Fit" with a specific coordinate preference, but applied to ALL bins 
    # and choosing the bin that yields the "best" (lowest) coordinate for the new box.
    # This is essentially "Best Fit" if we define "best" as "lowest position".
    
    def find_best_position(bin_boxes, w, h, d, W, H, D):
        """Find the position (x, y, z) in a bin that minimizes z, then y, then x,
        without overlapping existing boxes.
        Returns (x, y, z) or None.
        """
        # Generate candidate positions
        # 1. Origin
        candidates = [(0, 0, 0)]
        
        # 2. Surfaces of existing boxes
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Top face (higher z)
            candidates.append((bx, by, bz + bd))
            # Right face (higher x)
            candidates.append((bx + bw, by, bz))
            # Front face (higher y)
            candidates.append((bx, by + bh, bz))
            
            # Corners (often good for packing)
            candidates.append((bx + bw, by, bz + bd))
            candidates.append((bx, by + bh, bz + bd))
            candidates.append((bx + bw, by + bh, bz))
            candidates.append((bx, by, bz + bd))
            
            # Also consider positions aligned with the other axes
            # e.g. x aligned with bx, y aligned with by, z = bz + bd
            # This is covered by the surface generation above.
            
            # What about filling gaps?
            # If there is a box at (0,0,0) size (1,1,1) and another at (1,0,0) size (1,1,1),
            # a box of size (1,1,1) can go at (0,0,1) or (1,0,1) or (0,1,0) etc.
            # The surface generation covers (0,0,1) [top of first], (1,0,1) [top of second],
            # (1,0,0) [right of first], (0,1,0) [front of first].
            # It misses (1,1,0)? No, (1,1,0) is not a surface of a single box unless one is at (0,1,0) or (1,0,0).
            # Actually, the "corner" candidates cover many of these.
            
        # Deduplicate
        unique_candidates = list(set(candidates))
        
        best_pos = None
        best_cost = float('inf')
        
        for cx, cy, cz in unique_candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check overlap
            overlap = False
            for bx, by, bz, bw, bh, bd in bin_boxes:
                if (cx < bx + bw and bx < cx + w and
                    cy < by + bh and by < cy + h and
                    cz < bz + bd and bz < cz + d):
                    overlap = True
                    break
            
            if not overlap:
                cost = cz * 10000 + cy * 100 + cx
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (cx, cy, cz)
        
        return best_pos

    # Main packing loop
    bins = []  # List of lists of (x, y, z, w, h, d)
    placements = [None] * n
    
    for i, vol, w, h, d in indexed_boxes:
        best_bin_idx = -1
        best_pos = None
        best_global_cost = float('inf')
        
        # Try existing bins
        for b_idx in range(len(bins)):
            pos = find_best_position(bins[b_idx], w, h, d, W, H, D)
            if pos is not None:
                cx, cy, cz = pos
                cost = cz * 10000 + cy * 100 + cx
                # To implement "Best Fit" across bins, we might want to prefer bins that are "fuller"
                # or bins where the box fits "tighter".
                # For now, let's just pick the bin that allows the lowest position (most compact).
                if cost < best_global_cost:
                    best_global_cost = cost
                    best_bin_idx = b_idx
                    best_pos = pos
        
        if best_bin_idx == -1:
            # Open a new bin
            best_bin_idx = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        # Place the box
        bx, by, bz = best_pos
        bins[best_bin_idx].append((bx, by, bz, w, h, d))
        placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
    
    # Initial score
    current_score = evaluate_solution(placements)
    best_score = current_score
    best_placements = [tuple(p) for p in placements]
    
    # Local Search: Try to swap boxes between bins to reduce B or improve last bin fill
    # This is a structural addition: a repair/optimize phase.
    
    # Simple Local Search: 
    # 1. Identify the last bin (highest ID).
    # 2. Try to move boxes from the last bin to other bins.
    # 3. If a bin becomes empty, remove it (shift IDs).
    
    # We will do a few iterations of this.
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
            
        # Get current bin IDs
        bin_ids = set(p[0] for p in placements)
        if not bin_ids:
            break
        
        last_bin_id = max(bin_ids)
        other_bins = [b for b in bin_ids if b != last_bin_id]
        
        if not other_bins:
            break
            
        # Get boxes in last bin
        last_bin_boxes = [p for p in placements if p[0] == last_bin_id]
        
        # Try to move each box from last bin to another bin
        moved_any = False
        for box_idx, box_data in enumerate(last_bin_boxes):
            bin_id, x, y, z, w, h, d = box_data
            original_box_index = box_data[0] # Wait, placement tuple is (bin, x, y, z, w, h, d). 
            # We need the original index to update placements.
            # The placements list is indexed by original box index.
            # So box_data is placements[original_index].
            # Let's iterate by original index.
            
        # Redo iteration properly
        last_bin_box_indices = [i for i in range(n) if placements[i][0] == last_bin_id]
        
        for i in last_bin_box_indices:
            bin_id, x, y, z, w, h, d = placements[i]
            
            # Try to move box i to another bin
            for target_bin_id in other_bins:
                # Get boxes in target bin
                target_bin_boxes = [placements[j] for j in range(n) if placements[j][0] == target_bin_id]
                
                # Try to find a spot for (w, h, d) in target_bin_id
                # We need to temporarily remove the box from last bin? 
                # No, just check if it fits in target bin.
                # Note: The box is currently in last_bin. It doesn't overlap with target_bin.
                
                pos = find_best_position(target_bin_boxes, w, h, d, W, H, D)
                
                if pos is not None:
                    # We can move it!
                    # Update placements
                    placements[i] = (target_bin_id, pos[0], pos[1], pos[2], w, h, d)
                    moved_any = True
                    break # Stop trying other bins for this box
            
            if moved_any:
                # Re-evaluate score
                new_score = evaluate_solution(placements)
                if new_score < best_score:
                    best_score = new_score
                    best_placements = [tuple(p) for p in placements]
                # Restart the inner loop to check if more moves are possible
                break
        
        if not moved_any:
            # No more moves possible in this pass
            break

    return best_placements

# EVOLVE-BLOCK-END
