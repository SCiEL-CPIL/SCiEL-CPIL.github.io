"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations

def _orientations(w, h, d):
    """Return unique permutations of (w,h,d) as possible orientations."""
    perms = set(permutations((w, h, d)))
    return list(perms)

def _pack_2d_shelf(items, W, H):
    """
    2D shelf packing heuristic.
    items: list of (w, h) to pack into W x H area.
    Returns list of (x, y, w, h) for placed items, and list of indices not placed.
    """
    # Sort by height descending, then width descending
    indexed = list(enumerate(items))
    indexed.sort(key=lambda t: (-t[1][1], -t[1][0]))
    
    placements = {}  # index -> (x, y, w, h)
    current_x = 0
    current_y = 0
    shelf_height = 0
    
    for idx, (w, h) in indexed:
        # Check if we can fit in current shelf
        if current_x + w > W:
            # Move to next shelf
            current_y += shelf_height
            current_x = 0
            shelf_height = 0
        
        if current_y + h > H:
            # Can't fit in this layer
            continue
        
        # Place item
        placements[idx] = (current_x, current_y, w, h)
        current_x += w
        shelf_height = max(shelf_height, h)
    
    return placements

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Pre-compute all orientations for each box
    box_orients = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = _orientations(w, h, d)
        box_orients.append(orients)
    
    # Time-sliced layer decomposition: discretize container height into slabs
    # Strategy: try different slab heights and pick the best packing
    
    best_placements = None
    best_volume = 0.0
    
    # Generate candidate slab heights: unique heights from box orientations
    candidate_heights = set()
    for orients in box_orients:
        for ow, oh, od in orients:
            candidate_heights.add(oh)
    
    # Also try some heuristic slab heights
    if H > 0:
        candidate_heights.add(H // 2)
        candidate_heights.add(H // 3)
        candidate_heights.add(H // 4)
    
    # Filter valid slab heights
    candidate_heights = sorted([h for h in candidate_heights if 0 < h <= H], reverse=True)
    
    if not candidate_heights:
        candidate_heights = [H]
    
    for slab_h in candidate_heights:
        # Determine number of slabs
        num_slabs = H // slab_h
        if num_slabs == 0:
            continue
        
        # For each slab, pack boxes using 2D shelf heuristic
        # We'll try to assign boxes to slabs greedily
        
        # Create a list of (box_index, orientation) candidates
        # Sort boxes by volume descending for better packing
        box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        order = sorted(range(n), key=lambda i: -box_volumes[i])
        
        # Track which boxes are placed
        placed = [False] * n
        slab_placements = [[] for _ in range(num_slabs)]  # each slab: list of (idx, x, y, w, h, d)
        
        # For each box in order, try to place it in the first slab where it fits
        for idx in order:
            w, h, d = boxes[idx]
            placed_in_slab = False
            
            for slab_idx in range(num_slabs):
                # Try all orientations
                for ow, oh, od in box_orients[idx]:
                    if oh > slab_h:
                        continue
                    # Try to place in this slab using shelf heuristic
                    # We need to check if it fits in the current slab's 2D layout
                    # For simplicity, we'll use a simplified check:
                    # Check if the box fits in the remaining space of the slab
                    
                    # Get current placements in this slab
                    current_items = slab_placements[slab_idx]
                    
                    # Re-run 2D shelf packing with this box added to see if it fits
                    # This is expensive, so let's use a simpler heuristic:
                    # Check if there's enough "shelf space" remaining
                    
                    # Compute current shelf usage
                    current_x = 0
                    current_y = 0
                    shelf_height = 0
                    max_x_reached = 0
                    max_y_reached = 0
                    
                    for (_, px, py, pw, ph, pd) in current_items:
                        max_x_reached = max(max_x_reached, px + pw)
                        max_y_reached = max(max_y_reached, py + ph)
                    
                    # Simple check: can we fit the box in the W x slab_h area?
                    # We'll use a more careful check by attempting to place it
                    # in the shelf structure
                    
                    # Build the shelf structure from current items
                    shelves = {}  # shelf_y -> (max_x, shelf_height)
                    for (_, px, py, pw, ph, pd) in current_items:
                        if py not in shelves:
                            shelves[py] = [0, 0]
                        shelves[py][0] = max(shelves[py][0], px + pw)
                        shelves[py][1] = max(shelves[py][1], ph)
                    
                    # Try to add the box to an existing shelf or a new shelf
                    found_spot = False
                    for shelf_y, (shelf_max_x, shelf_h) in shelves.items():
                        if shelf_y + oh <= slab_h and shelf_max_x + ow <= W:
                            # Can add to this shelf
                            new_x = shelf_max_x
                            new_y = shelf_y
                            # Check overlap with existing items in this shelf
                            overlap = False
                            for (_, px, py, pw, ph, pd) in current_items:
                                if py == shelf_y:
                                    if not (new_x + ow <= px or px + pw <= new_x):
                                        if not (new_y + oh <= py or py + ph <= new_y):
                                            overlap = True
                                            break
                            if not overlap:
                                slab_placements[slab_idx].append((idx, new_x, new_y, ow, oh, od))
                                placed_in_slab = True
                                found_spot = True
                                break
                        # Try a new shelf below
                        # Find the lowest available y position
                        # This is complex; let's try a simple approach:
                        # Try y=0, then y=shelf_h for each existing shelf
                        pass
                    
                    if not found_spot:
                        # Try to place in a new shelf at the bottom
                        # Find the minimum y where we can place a new shelf
                        min_y = 0
                        for shelf_y, (shelf_max_x, shelf_h) in shelves.items():
                            if shelf_y > min_y:
                                min_y = shelf_y  # This isn't quite right
                        # Actually, let's just try y=0 first
                        if 0 + oh <= slab_h and 0 + ow <= W:
                            # Check if y=0 is free
                            overlap = False
                            for (_, px, py, pw, ph, pd) in current_items:
                                if not (0 + ow <= px or px + pw <= 0):
                                    if not (0 + oh <= py or py + ph <= 0):
                                        overlap = True
                                        break
                            if not overlap:
                                slab_placements[slab_idx].append((idx, 0, 0, ow, oh, od))
                                placed_in_slab = True
                                found_spot = True
                    
                    if found_spot:
                        break
                
                if placed_in_slab:
                    break
            
            if placed_in_slab:
                placed[idx] = True
        
        # Calculate total volume placed
        total_volume = 0.0
        for slab_idx in range(num_slabs):
            for (idx, x, y, w, h, d) in slab_placements[slab_idx]:
                total_volume += w * h * d
        
        if total_volume > best_volume:
            best_volume = total_volume
            # Store the best placement
            best_placements = [None] * n
            for slab_idx in range(num_slabs):
                for (idx, x, y, w, h, d) in slab_placements[slab_idx]:
                    best_placements[idx] = (0, x, y, slab_idx * slab_h, w, h, d)
    
    # Build final result
    result = []
    for i in range(n):
        if best_placements is not None and best_placements[i] is not None:
            result.append(best_placements[i])
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
