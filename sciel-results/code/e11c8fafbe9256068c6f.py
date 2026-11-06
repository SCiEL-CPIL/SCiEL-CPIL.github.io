import time
import random
import itertools
import math
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    # Safety margin: The evaluator kills at 150s, but the task says 120s limit.
    # We use 110s to be safe.
    deadline = time.time() + 110.0
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # Grid Resolution:
    # We need a balance between resolution (ability to fit boxes) and performance (grid size).
    # If dimensions are large, a fine grid is too slow.
    # Target grid size ~ 100x100x100 = 1M cells (manageable in NumPy).
    max_dim = max(W, H, D)
    if max_dim > 500:
        res = 10
    elif max_dim > 200:
        res = 5
    else:
        res = 2
        
    # Ensure res is at least 1
    res = max(1, res)
    
    # Grid dimensions (number of cells)
    GW = W // res
    GH = H // res
    GD = D // res
    
    # If grid is too small or large, adjust
    if GW < 5 or GH < 5 or GD < 5:
        # Fallback to a coarser logic or just use 1
        res = 1
        GW = W
        GH = H
        GD = D
        
    # If grid is huge, increase res
    while GW * GH * GD > 2000000:
        res *= 2
        GW = W // res
        GH = H // res
        GD = D // res

    # Initialize occupancy grid
    # 0: empty, 1: occupied
    grid = np.zeros((GW, GH, GD), dtype=np.int8)
    
    # Helper: Check if a box fits at grid position (gx, gy, gz) with dims (bw, bh, bd) in grid units
    # Returns True if fits, False otherwise. Does not modify grid.
    def fits(gx, gy, gz, bw, bh, bd):
        if gx + bw > GW or gy + bh > GH or gz + bd > GD:
            return False
        # Check occupancy
        # We need to check the sub-grid
        sub = grid[gx:gx+bw, gy:gy+bh, gz:gz+bd]
        if np.any(sub > 0):
            return False
        return True

    # Helper: Place box in grid
    def place(gx, gy, gz, bw, bh, bd):
        grid[gx:gx+bw, gy:gy+bh, gz:gz+bd] = 1

    # Helper: Remove box from grid
    def remove(gx, gy, gz, bw, bh, bd):
        grid[gx:gx+bw, gy:gy+bh, gz:gz+bd] = 0

    # Convert box dims to grid units
    # Note: We must be careful with rounding. 
    # If a box is 3 units and res=2, it occupies 2 grid cells (4 units) or 1?
    # To guarantee no overlap in physical space, we should round UP grid dimensions.
    # However, this might make boxes "too big" for the grid.
    # Alternative: Use exact integer coordinates in the search, but use the grid for fast collision checks?
    # Given the 120s limit and complexity, a pure grid approach with "snapping" is safer.
    # We will snap box positions to grid points and size them to grid cells.
    # This ensures integer coordinates in output, which is usually preferred or at least valid.
    # The "w, h, d" in output should be the original box dimensions? 
    # The task says: "places a subset of boxes... with bin 0 for loaded boxes".
    # Usually, the dimensions in the output should match the box's dimensions in that orientation.
    # If we snap to grid, the physical box might be smaller than the grid cell.
    # This is safe (no overlap) but might leave gaps.
    # To maximize volume, we want tight packing.
    # Let's try to use the grid only for *collision detection* but keep positions as integers 
    # derived from grid * res.
    
    # Revised Plan:
    # 1. Grid is for fast collision checking.
    # 2. Positions are multiples of `res`.
    # 3. Box dimensions are the original box dimensions (integers).
    # 4. When checking `fits`, we convert box dims to grid units (ceil) to ensure safety.
    
    def dims_to_grid(w, h, d):
        return (math.ceil(w/res), math.ceil(h/res), math.ceil(d/res))

    def check_fits_exact(gx, gy, gz, w, h, d):
        # gx, gy, gz are grid indices. Physical pos: gx*res, gy*res, gz*res
        # Physical box size: w, h, d
        # Grid cells covered:
        gw, gh, gd = dims_to_grid(w, h, d)
        if gx + gw > GW or gy + gh > GH or gz + gd > GD:
            return False
        sub = grid[gx:gx+gw, gy:gy+gh, gz:gz+gd]
        return not np.any(sub > 0)

    def place_exact(gx, gy, gz, w, h, d):
        gw, gh, gd = dims_to_grid(w, h, d)
        grid[gx:gx+gw, gy:gy+gh, gz:gz+gd] = 1

    def remove_exact(gx, gy, gz, w, h, d):
        gw, gh, gd = dims_to_grid(w, h, d)
        grid[gx:gx+gw, gy:gy+gh, gz:gz+gd] = 0

    # State representation:
    # placed: list of dicts {idx, gx, gy, gz, w, h, d}
    # We will maintain a list of placed boxes.
    
    def greedy_pack(order):
        placed = []
        # Reset grid
        grid[:] = 0
        total_vol = 0
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            # Search for best position
            # To speed up, we can iterate over grid cells.
            # But GW*GH*GD can be large.
            # Optimization: Only check cells where the box *could* fit?
            # For now, iterate all valid starting positions.
            
            # Pre-compute grid dims for this box orientation
            # We try all orientations
            
            for (bw, bh, bd) in orient_cache[idx]:
                gw, gh, gd = dims_to_grid(bw, bh, bd)
                
                # Iterate possible positions
                # We can limit the search space. 
                # A simple heuristic: scan from bottom-up, left-right, front-back.
                # First fit decreasing style.
                for gz in range(GD - gd + 1):
                    for gy in range(GH - gh + 1):
                        for gx in range(GW - gw + 1):
                            if check_fits_exact(gx, gy, gz, bw, bh, bd):
                                # Score: prefer lower z, then lower y, then lower x (compactness)
                                # Or maximize contact? 
                                # For speed, simple first-fit (lowest z, y, x) is fast.
                                # Let's use a score that encourages filling gaps:
                                # Score = z*10000 + y*100 + x
                                score = gz * 10000 + gy * 100 + gx
                                if score < best_score:
                                    best_score = score
                                    best_pos = (gx, gy, gz, bw, bh, bd)
                                # Break early if we find the "perfect" bottom-left-front?
                                # If gz==0, gy==0, gx==0, it's the best possible for this orientation.
                                if gz == 0 and gy == 0 and gx == 0:
                                    break
                        if best_pos and best_pos[2] == 0 and best_pos[1] == 0 and best_pos[0] == 0:
                            break
                    if best_pos and best_pos[2] == 0 and best_pos[1] == 0 and best_pos[0] == 0:
                        break
                if best_pos:
                    # We found a spot for this orientation. 
                    # Should we check other orientations? 
                    # Greedy: take the first valid orientation that fits? 
                    # Or the one that fits best? 
                    # For speed, let's take the first valid orientation that fits in the "lowest" position.
                    # Actually, the loop above iterates orientations. 
                    # We should pick the best (gx,gy,gz) across ALL orientations.
                    # Let's restructure: 
                    # 1. Iterate all orientations.
                    # 2. For each, find best pos.
                    # 3. Pick global best.
                    pass # Placeholder, logic below
                    
            # Refactored Greedy Step:
            best_global_pos = None
            best_global_score = float('inf')
            for (bw, bh, bd) in orient_cache[idx]:
                gw, gh, gd = dims_to_grid(bw, bh, bd)
                if gw > GW or gh > GH or gd > GD:
                    continue
                # Scan grid
                # Optimization: If grid is large, this is slow.
                # Use numpy to find empty spots?
                # For now, linear scan.
                for gz in range(GD - gd + 1):
                    for gy in range(GH - gh + 1):
                        for gx in range(GW - gw + 1):
                            if check_fits_exact(gx, gy, gz, bw, bh, bd):
                                score = gz * 10000 + gy * 100 + gx
                                if score < best_global_score:
                                    best_global_score = score
                                    best_global_pos = (gx, gy, gz, bw, bh, bd)
                                # Early exit if perfect
                                if gz == 0 and gy == 0 and gx == 0:
                                    break
                        if best_global_pos and best_global_pos[2] == 0 and best_global_pos[1] == 0 and best_global_pos[0] == 0:
                            break
                    if best_global_pos and best_global_pos[2] == 0 and best_global_pos[1] == 0 and best_global_pos[0] == 0:
                        break
                
            if best_global_pos:
                gx, gy, gz, bw, bh, bd = best_global_pos
                place_exact(gx, gy, gz, bw, bh, bd)
                placed.append({
                    'idx': idx,
                    'gx': gx, 'gy': gy, 'gz': gz,
                    'w': bw, 'h': bh, 'd': bd
                })
                total_vol += bw * bh * bd
                
        return placed, total_vol

    # Initial Orders
    indexed = list(range(n))
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    
    all_orders = [order_vol, order_max]
    for _ in range(3):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        all_orders.append(rand_order)
        
    best_placed = []
    best_vol = 0
    
    # Run Greedy on initial orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p, v = greedy_pack(order)
        if v > best_vol:
            best_vol = v
            best_placed = p[:]
            
    # Simulated Annealing on Coordinates
    # We take the best greedy solution and try to improve it.
    # Move: Pick a placed box, try to move it to a nearby empty spot.
    # Or: Pick a placed box, try to rotate it.
    # Or: Pick a placed box, remove it, and re-insert it optimally (Reinsertion).
    # Reinsertion is powerful but expensive.
    # Let's try "Shift": Move a box to an adjacent empty grid cell.
    
    if best_placed and time.time() < deadline:
        current_placed = best_placed[:]
        current_vol = best_vol
        # Rebuild grid for current state
        grid[:] = 0
        for b in current_placed:
            place_exact(b['gx'], b['gy'], b['gz'], b['w'], b['h'], b['d'])
            
        T = 50.0
        T_min = 0.1
        cooling = 0.99
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve < max_no_improve:
            if not current_placed:
                break
                
            # Pick a random box
            b_idx = random.randint(0, len(current_placed) - 1)
            box = current_placed[b_idx]
            idx = box['idx']
            gx, gy, gz = box['gx'], box['gy'], box['gz']
            bw, bh, bd = box['w'], box['h'], box['d']
            
            # Remove box
            remove_exact(gx, gy, gz, bw, bh, bd)
            
            # Try to find a new spot
            # Strategy: Look for a spot near the current one, or the best global spot?
            # Global best is expensive. Local search is faster.
            # Let's try a random valid spot from a subset of candidates.
            # Candidates: All grid cells? Too many.
            # Let's try "First Fit Decreasing" style scan but stop early?
            # Or just try a few random positions?
            
            # Better: Try to re-insert using the same greedy logic but limited?
            # Actually, let's try moving to a "better" score position.
            # Score = z*10000 + y*100 + x. Lower is better.
            # If we can find a position with lower score, it's an improvement in compactness.
            # But volume is constant. We want to enable *other* boxes to fit?
            # No, we are moving an already placed box. Volume doesn't change.
            # So how does this help? 
            # It doesn't directly increase volume. 
            # It helps if we allow *unplaced* boxes to be placed later?
            # No, the set of placed boxes is fixed in this SA phase.
            
            # CRITICAL REALIZATION: 
            # Moving placed boxes doesn't change the volume of the *current* packing.
            # To increase volume, we must:
            # 1. Place an unplaced box.
            # 2. Or, rearrange placed boxes to create space for an unplaced box.
            
            # So the move should be: 
            # 1. Pick a placed box B.
            # 2. Remove B.
            # 3. Try to place an unplaced box U in the space freed or elsewhere.
            # 4. If U fits, we gain volume.
            # 5. Then try to re-place B in the remaining space.
            # If B fits back, we have improved volume.
            
            # This is a "Swap/Insert" move.
            
            # Let's identify unplaced boxes
            placed_idxs = set(b['idx'] for b in current_placed)
            unplaced_idxs = [i for i in range(n) if i not in placed_idxs]
            
            if not unplaced_idxs:
                break # All placed, perfect pack
            
            # Pick a random unplaced box
            u_idx = random.choice(unplaced_idxs)
            u_w, u_h, u_d = boxes[u_idx]
            
            # Pick a random placed box to move
            b_idx = random.randint(0, len(current_placed) - 1)
            box = current_placed[b_idx]
            gx, gy, gz = box['gx'], box['gy'], box['gz']
            bw, bh, bd = box['w'], box['h'], box['d']
            
            # Remove B
            remove_exact(gx, gy, gz, bw, bh, bd)
            
            # Try to place U
            u_placed = False
            u_pos = None
            for (uw, uh, ud) in orient_cache[u_idx]:
                gw, gh, gd = dims_to_grid(uw, uh, ud)
                if gw > GW or gh > GH or gd > GD:
                    continue
                # Scan for U
                # To speed up, we can just check if it fits anywhere.
                # We don't need the *best* spot for U, just *any* spot, 
                # as long as B can also be placed.
                # But finding *any* spot is still O(GW*GH*GD).
                # Let's use a quick check: Is there space?
                # For speed, let's just try the "greedy best" spot for U.
                best_u_pos = None
                best_u_score = float('inf')
                for uz in range(GD - gd + 1):
                    for uy in range(GH - gh + 1):
                        for ux in range(GW - gw + 1):
                            if check_fits_exact(ux, uy, uz, uw, uh, ud):
                                score = uz * 10000 + uy * 100 + ux
                                if score < best_u_score:
                                    best_u_score = score
                                    best_u_pos = (ux, uy, uz, uw, uh, ud)
                                if uz == 0 and uy == 0 and ux == 0:
                                    break
                        if best_u_pos and best_u_pos[2] == 0 and best_u_pos[1] == 0 and best_u_pos[0] == 0:
                            break
                    if best_u_pos and best_u_pos[2] == 0 and best_u_pos[1] == 0 and best_u_pos[0] == 0:
                        break
                
                if best_u_pos:
                    # U fits. Now try to place B back.
                    # We need to place B in the space remaining after U is placed.
                    # Tentatively place U
                    ux, uy, uz, uw, uh, ud = best_u_pos
                    place_exact(ux, uy, uz, uw, uh, ud)
                    
                    # Try to place B
                    b_placed = False
                    for (bw2, bh2, bd2) in orient_cache[box['idx']]:
                        gw2, gh2, gd2 = dims_to_grid(bw2, bh2, bd2)
                        if gw2 > GW or gh2 > GH or gd2 > GD:
                            continue
                        best_b_pos = None
                        best_b_score = float('inf')
                        for bz in range(GD - gd2 + 1):
                            for by in range(GH - gh2 + 1):
                                for bx in range(GW - gw2 + 1):
                                    if check_fits_exact(bx, by, bz, bw2, bh2, bd2):
                                        score = bz * 10000 + by * 100 + bx
                                        if score < best_b_score:
                                            best_b_score = score
                                            best_b_pos = (bx, by, bz, bw2, bh2, bd2)
                                        if bz == 0 and by == 0 and bx == 0:
                                            break
                            if best_b_pos and best_b_pos[2] == 0 and best_b_pos[1] == 0 and best_b_pos[0] == 0:
                                break
                        if best_b_pos and best_b_pos[2] == 0 and best_b_pos[1] == 0 and best_b_pos[0] == 0:
                            break
                        
                        if best_b_pos:
                            # B fits!
                            b_placed = True
                            break
                    
                    if b_placed:
                        # Success! We swapped U in and B out? 
                        # No, we want to keep B and add U? 
                        # Wait, the move is: Remove B, Place U, Place B.
                        # If B fits back, we have added U to the packing.
                        # Volume increases by vol(U).
                        u_placed = True
                        # Update state
                        # U is now placed at best_u_pos
                        # B is now placed at best_b_pos
                        # We need to update current_placed list
                        # Remove old B entry
                        current_placed.pop(b_idx)
                        # Add U
                        current_placed.append({
                            'idx': u_idx,
                            'gx': ux, 'gy': uy, 'gz': uz,
                            'w': uw, 'h': uh, 'd': ud
                        })
                        # Add B
                        bx, by, bz, bw2, bh2, bd2 = best_b_pos
                        current_placed.append({
                            'idx': box['idx'],
                            'gx': bx, 'gy': by, 'gz': bz,
                            'w': bw2, 'h': bh2, 'd': bd2
                        })
                        current_vol += box_vols[u_idx]
                        if current_vol > best_vol:
                            best_vol = current_vol
                            best_placed = current_placed[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # B doesn't fit back. Undo U.
                        remove_exact(ux, uy, uz, uw, uh, ud)
                        # Re-place B at original spot? 
                        # Or try to place B somewhere else?
                        # If we don't re-place B, we lose volume.
                        # Let's just re-place B at original spot to maintain state.
                        place_exact(gx, gy, gz, bw, bh, bd)
                        current_placed[b_idx] = box # Restore
                        no_improve += 1
                    break # Only try first orientation for U? 
                    # Actually, we should try all orientations for U to find *any* fit.
                    # The loop above breaks on first fit. 
                    # This is a valid "Greedy Insertion" move.
                    
            if u_placed:
                # If we successfully inserted U, we are done with this move.
                pass
            else:
                # If we failed to insert U, we restored B.
                pass
                
            T *= cooling
            if T < T_min:
                T = T_min
                
    # Finalize result
    result = [None] * n
    for p in best_placed:
        idx = p['idx']
        x = p['gx'] * res
        y = p['gy'] * res
        z = p['gz'] * res
        w, h, d = p['w'], p['h'], p['d']
        result[idx] = (0, x, y, z, w, h, d)
        
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result

# EVOLVE-BLOCK-END
