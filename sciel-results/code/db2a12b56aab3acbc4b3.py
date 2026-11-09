"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Precompute all 6 rotations for each box
    def get_rotations(w, h, d):
        return [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
    
    # Sort boxes by volume descending
    box_volumes = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: -box_volumes[i])
    
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Layered Shelf Structure
    # layers: list of dicts
    # Each layer: { 'z': start_z, 'thickness': max_d_in_layer, 'rows': [] }
    # Each row: { 'y': start_y, 'height': max_h_in_row, 'x': current_x, 'boxes': [] }
    
    layers = []
    
    def try_place_in_layer(layer, w, h, d, box_idx):
        """
        Try to place a box with dims (w, h, d) in the given layer.
        w is along X, h is along Y, d is the layer thickness (Z).
        Returns True if placed, False otherwise.
        If placed, updates the layer structure.
        """
        # Try existing rows
        for row in layer['rows']:
            # Check if box fits vertically in this row's available space
            # The row starts at row['y']. The box has height h.
            # It must fit within H.
            if row['y'] + h <= H:
                # Check if it fits horizontally
                if row['x'] + w <= W:
                    # Place it
                    x = row['x']
                    y = row['y']
                    z = layer['z']
                    
                    # Update row
                    row['x'] += w
                    row['height'] = max(row['height'], h)
                    row['boxes'].append(box_idx)
                    
                    # Update layer thickness (d is the depth of the box, which is the Z-dimension used)
                    # Wait, in this model, 'd' is the dimension along Z.
                    # The layer thickness is the max Z-dimension of boxes in this layer.
                    layer['thickness'] = max(layer['thickness'], d)
                    
                    # Record placement
                    # The box is placed at (x, y, z) with dimensions (w, h, d)
                    # Note: The rotation (w, h, d) corresponds to (X, Y, Z)
                    placements[box_idx] = (0, x, y, z, w, h, d)
                    return True
            
            # If it doesn't fit in the current row (x overflow), we could start a new row
            # But we should check if a new row is possible before giving up on the layer?
            # Standard shelf: if x overflows, move to next row.
            # But we are iterating rows. If the first row is full, we check the second.
            # Actually, in a shelf, items are placed sequentially.
            # If row['x'] + w > W, it doesn't fit in THIS row.
            # We should check if we can start a NEW row in this layer.
            # We handle "new row" logic outside the loop or at the end?
            # Better: Check if we can fit in ANY existing row.
            # If not, check if we can start a NEW row in this layer.
        
        # If no existing row worked, try to start a new row in this layer
        if not layer['rows']:
            new_y = 0
        else:
            last_row = layer['rows'][-1]
            new_y = last_row['y'] + last_row['height']
            
        if new_y + h <= H:
            # Create new row
            new_row = {
                'y': new_y,
                'height': h,
                'x': w, # Box is placed at x=0 of this new row
                'boxes': [box_idx]
            }
            layer['rows'].append(new_row)
            layer['thickness'] = max(layer['thickness'], d)
            
            # Record placement
            x = 0
            y = new_y
            z = layer['z']
            placements[box_idx] = (0, x, y, z, w, h, d)
            return True
            
        return False

    def try_place_box(box_idx):
        w0, h0, d0 = boxes[box_idx]
        rots = get_rotations(w0, h0, d0)
        
        # We want to place the box in the lowest possible Z (layer)
        # And within that layer, the lowest possible Y (row)
        
        # Try each rotation
        # We prefer rotations that minimize the "footprint" or fit best?
        # Let's just try all rotations and pick the first one that fits in the lowest layer.
        # To be more optimal, we can score the placements.
        # For now, let's try to find the best (lowest Z, then lowest Y, then lowest X) position.
        
        best_placement = None
        best_score = float('inf')
        
        for rot in rots:
            rw, rh, rd = rot
            
            # Try existing layers (sorted by Z)
            # We should iterate layers in order of Z
            for layer in layers:
                # Check if the box fits in the Z dimension of this layer
                # The layer starts at layer['z']. The box has depth rd.
                # It must fit in the container: layer['z'] + rd <= D
                # Also, it must fit in the layer's allocated thickness?
                # No, the layer thickness grows as we add boxes.
                # But we must ensure that if we add this box, the layer doesn't exceed D?
                # Actually, the layer thickness is the max depth of boxes in it.
                # So we need: layer['z'] + max(layer['thickness'], rd) <= D
                # And for the 2D part:
                # The box (rw, rh) must fit in the W x H area starting from (0, layer_y_offset?)
                # No, the 2D area is W x H for the whole layer.
                
                # Check Z constraint
                new_thickness = max(layer['thickness'], rd)
                if layer['z'] + new_thickness > D:
                    continue
                
                # Try to place in this layer
                # We need to simulate the placement to see if it fits and what the position is.
                # My try_place_in_layer modifies state. We need a dry-run or a way to rollback.
                # For simplicity, let's just try to place it. If it works, we accept it.
                # But we want the BEST placement among rotations and layers.
                # This greedy "first fit" might be suboptimal.
                # Let's stick to a simpler greedy: 
                # 1. Iterate layers in order of Z.
                # 2. For the first layer where it fits (Z-wise), try to place it.
                # 3. If it places, accept and break.
                # 4. If it doesn't fit in any existing layer, create a new layer.
                
                # To handle "best fit" within a layer, we can check if it fits in an existing row.
                # If yes, place it there.
                # If no, check if a new row fits.
                # If yes, place it in new row.
                # If no, try next layer.
                
                # Let's refine try_place_in_layer to return the position or None.
                # And not modify state until we are sure.
                # This requires more code.
                
                # Alternative: Just use the first-fit strategy.
                # It's simpler and often good enough.
                pass # Placeholder for logic below

            # If no existing layer worked, create a new layer
            # Find the highest Z
            if not layers:
                new_z = 0
            else:
                # Layers are not necessarily sorted by Z if we just append?
                # We should keep layers sorted or find the max Z.
                max_z = max(l['z'] + l['thickness'] for l in layers)
                new_z = max_z
            
            if new_z + rd <= D:
                # Create new layer
                new_layer = {
                    'z': new_z,
                    'thickness': rd,
                    'rows': []
                }
                layers.append(new_layer)
                
                # Place in this new layer (it will be the first row)
                # We can call try_place_in_layer on the new layer
                # But we need to ensure it's the only box so far.
                # Let's just place it directly.
                x = 0
                y = 0
                z = new_z
                
                new_layer['rows'].append({
                    'y': 0,
                    'height': rh,
                    'x': rw,
                    'boxes': [box_idx]
                })
                
                placements[box_idx] = (0, x, y, z, rw, rh, rd)
                return True
        
        return False

    # The logic above is a bit messy with the "try" vs "do" and best-fit.
    # Let's rewrite the core loop to be cleaner.
    
    # Clear layers and placements for the new logic
    layers = []
    for i in range(n):
        placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    def get_best_position_for_rot(rot, layer):
        """
        Returns (x, y, z, success) for a box with dims rot in a specific layer.
        Does NOT modify state.
        """
        rw, rh, rd = rot
        
        # Check Z fit
        new_thickness = max(layer['thickness'], rd)
        if layer['z'] + new_thickness > D:
            return None
            
        # Check 2D fit in existing rows
        for row in layer['rows']:
            if row['y'] + rh <= H:
                if row['x'] + rw <= W:
                    return (row['x'], row['y'], layer['z'], True)
        
        # Check 2D fit in new row
        if not layer['rows']:
            new_y = 0
        else:
            last_row = layer['rows'][-1]
            new_y = last_row['y'] + last_row['height']
            
        if new_y + rh <= H:
            return (0, new_y, layer['z'], True)
            
        return None

    for box_idx in sorted_indices:
        w0, h0, d0 = boxes[box_idx]
        rots = get_rotations(w0, h0, d0)
        
        placed = False
        best_pos = None
        best_layer_idx = -1
        best_score = float('inf')
        
        # Try all rotations
        for rot in rots:
            rw, rh, rd = rot
            
            # Try existing layers
            for i, layer in enumerate(layers):
                pos = get_best_position_for_rot(rot, layer)
                if pos:
                    x, y, z, _ = pos
                    # Score: prefer lower Z, then lower Y, then lower X
                    score = z * W * H + y * W + x
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z, rw, rh, rd)
                        best_layer_idx = i
            
            # Try new layer
            # Calculate new Z
            if not layers:
                new_z = 0
            else:
                max_z = max(l['z'] + l['thickness'] for l in layers)
                new_z = max_z
            
            if new_z + rd <= D:
                # It fits in a new layer at (0,0,new_z)
                x, y, z = 0, 0, new_z
                score = z * W * H + y * W + x
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z, rw, rh, rd)
                    best_layer_idx = -2 # Special marker for new layer
        
        if best_pos:
            x, y, z, rw, rh, rd = best_pos
            placements[box_idx] = (0, x, y, z, rw, rh, rd)
            
            # Update state
            if best_layer_idx == -2:
                # Create new layer
                new_layer = {
                    'z': z,
                    'thickness': rd,
                    'rows': []
                }
                layers.append(new_layer)
                # Add to new row
                new_layer['rows'].append({
                    'y': y,
                    'height': rh,
                    'x': x + rw, # x is 0, so x+rw is rw
                    'boxes': [box_idx]
                })
            else:
                layer = layers[best_layer_idx]
                # Check if it was placed in an existing row or a new row
                # We need to know which row.
                # get_best_position_for_rot returned the position.
                # We can find the row by y and x?
                # Or we can re-evaluate.
                # Let's just find the row that matches y and check if x was the end.
                
                placed_in_existing = False
                for row in layer['rows']:
                    if row['y'] == y and row['x'] == x:
                        # It was placed at the end of this row
                        row['x'] += rw
                        row['height'] = max(row['height'], rh)
                        row['boxes'].append(box_idx)
                        layer['thickness'] = max(layer['thickness'], rd)
                        placed_in_existing = True
                        break
                
                if not placed_in_existing:
                    # It must be a new row
                    new_row = {
                        'y': y,
                        'height': rh,
                        'x': x + rw,
                        'boxes': [box_idx]
                    }
                    layer['rows'].append(new_row)
                    layer['thickness'] = max(layer['thickness'], rd)
            
            placed = True
            
        if not placed:
            # Box left out
            pass

    return placements
# EVOLVE-BLOCK-END
