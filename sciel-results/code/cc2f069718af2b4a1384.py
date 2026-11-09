import time
import random
import math
from collections import defaultdict

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute unique rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        # Sort rotations to have a deterministic order, often putting larger dims first or similar
        box_rotations.append(list(rots))

    # Data structures for the current solution
    # placed_boxes: list of [box_id, x, y, z, w, h, d]
    
    def check_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True

    def can_place(placed_boxes, x, y, z, w, h, d):
        """Check if a box can be placed at (x,y,z) with dimensions (w,h,d)."""
        if x + w > W or y + h > H or z + d > D:
            return False
        
        # Simple O(N) check is often fast enough for N=60 if we prune early
        # But for speed in local search, we might need spatial indexing.
        # Given N=60, O(N) is 60 checks. With 115s, we can afford millions of checks.
        # Let's stick to O(N) for simplicity and reliability, but optimize the inner loop.
        
        for pb in placed_boxes:
            _, px, py, pz, pw, ph, pd = pb
            if check_overlap(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def greedy_place(order, strategy='best_fit'):
        """
        Place boxes in the given order.
        strategy: 'first_fit' or 'best_fit'
        """
        placed_boxes = []
        placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
        # Candidate positions: start with (0,0,0)
        # We will generate new candidates based on placed boxes
        candidates = [(0, 0, 0)]
        # To avoid duplicates in candidates, we could use a set, but list is fine if we deduplicate logic
        # Actually, maintaining a list of unique candidates is important for performance.
        # Let's use a set for candidate generation to keep it clean, then convert to list for iteration
        candidate_set = {(0, 0, 0)}
        
        for box_id in order:
            w_orig, h_orig, d_orig = boxes[box_id]
            
            found_placement = False
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            # Try all rotations
            for rot in box_rotations[box_id]:
                rw, rh, rd = rot
                
                # Try all candidate positions
                # We iterate over a snapshot of the candidate set
                # Note: candidates can grow. We should probably limit or prune if it gets too large.
                # For 60 boxes, candidates usually stay manageable.
                
                for cx, cy, cz in candidate_set:
                    if can_place(placed_boxes, cx, cy, cz, rw, rh, rd):
                        # Evaluate score
                        # Heuristic: Prefer positions that are "low" and "tight"
                        # Minimizing x+y+z tends to pack towards the origin
                        score = cx + cy + cz
                        
                        if strategy == 'first_fit':
                            # Just take the first one found in this rotation/candidate loop
                            # This is faster but less optimal
                            if not found_placement:
                                best_pos = (cx, cy, cz)
                                best_rot = rot
                                found_placement = True
                                break # Break candidate loop for this rotation? Or break all?
                                # Let's break candidate loop, but continue to next rotation?
                                # No, first fit usually means first valid position overall.
                                # Let's stick to: find first valid (rot, pos) pair.
                                # To make it more robust, let's collect all valids for 'best_fit' and break early for 'first_fit'.
                        
                        if strategy == 'best_fit':
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = rot
                
                if strategy == 'first_fit' and found_placement:
                    break
            
            if best_pos:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed_boxes.append([box_id, bx, by, bz, bw, bh, bd])
                placements[box_id] = (0, bx, by, bz, bw, bh, bd)
                
                # Generate new candidate positions based on the newly placed box
                # New corners are:
                # (x+w, y, z), (x+w, y+h, z), (x+w, y, z+d), (x+w, y+h, z+d)
                # (x, y+h, z), (x, y+h, z+d)
                # (x, y, z+d)
                
                new_corners = [
                    (bx + bw, by, bz),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                    (bx, by + bh, bz),
                    (bx, by + bh, bz + bd),
                    (bx, by, bz + bd)
                ]
                
                for nc in new_corners:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        candidate_set.add(nc)
            
            # Optional: Prune candidate set if it gets too large
            # If len(candidate_set) > 10000, we might want to prune, but for N=60 it's rarely an issue.
            
        vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        return placements, vol, placed_boxes

    def local_search(placements, placed_boxes, max_iter=500):
        """
        Improve the solution by moving boxes to better positions.
        """
        current_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        
        # We will try to move each placed box to a "better" position
        # A better position is one that allows other boxes to be placed or reduces the "bounding box" of used space.
        # Simple heuristic: Try to move a box to a position closer to (0,0,0) if valid.
        # Or, try to shift boxes to fill gaps.
        
        # Given the time constraint, a simple "shake" or "move to lower corner" might be enough.
        # Let's try: For each box, try to move it to a valid position with lower (x+y+z).
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Shuffle order to avoid bias
            box_ids_in_order = [pb[0] for pb in placed_boxes]
            random.shuffle(box_ids_in_order)
            
            for box_id in box_ids_in_order:
                if time.time() > deadline - 5:
                    break
                
                # Find current placement
                idx = None
                for i, p in enumerate(placements):
                    if p[0] == box_id:
                        idx = i
                        break
                if idx is None:
                    continue
                
                _, cx, cy, cz, cw, ch, cd = placements[box_id]
                
                # Temporarily remove box
                placed_boxes_copy = [pb for pb in placed_boxes if pb[0] != box_id]
                
                # Find best new position for this box
                # We can reuse the candidate set logic, but for a single box, we can just scan a grid or use existing candidates.
                # For simplicity and speed, let's try shifting by small amounts or re-evaluating against a grid of "free" spots.
                # Actually, the most effective local search for this is "Swap" or "Move".
                # Let's try moving to any valid position that is "better" (lower x+y+z).
                
                # To avoid O(N^2) candidate generation for every move, we can just try a few specific moves:
                # 1. Shift to left (x-1) if valid
                # 2. Shift to back (y-1) if valid
                # 3. Shift to down (z-1) if valid
                # This is a very simple gradient descent.
                
                best_new_pos = None
                best_score = cx + cy + cz
                current_score = best_score
                
                # Try moving to lower x
                if cx > 0:
                    if can_place(placed_boxes_copy, cx-1, cy, cz, cw, ch, cd):
                        new_score = (cx-1) + cy + cz
                        if new_score < best_score:
                            best_score = new_score
                            best_new_pos = (cx-1, cy, cz)
                
                # Try moving to lower y
                if cy > 0:
                    if can_place(placed_boxes_copy, cx, cy-1, cz, cw, ch, cd):
                        new_score = cx + (cy-1) + cz
                        if new_score < best_score:
                            best_score = new_score
                            best_new_pos = (cx, cy-1, cz)
                
                # Try moving to lower z
                if cz > 0:
                    if can_place(placed_boxes_copy, cx, cy, cz-1, cw, ch, cd):
                        new_score = cx + cy + (cz-1)
                        if new_score < best_score:
                            best_score = new_score
                            best_new_pos = (cx, cy, cz-1)
                
                # If we found a better position, move the box
                if best_new_pos:
                    nx, ny, nz = best_new_pos
                    # Update placements
                    placements[box_id] = (0, nx, ny, nz, cw, ch, cd)
                    # Update placed_boxes
                    for i, pb in enumerate(placed_boxes):
                        if pb[0] == box_id:
                            placed_boxes[i] = [box_id, nx, ny, nz, cw, ch, cd]
                            break
                    improved = True
        
        # Recalculate volume
        final_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        return placements, final_vol, placed_boxes

    # Main Search Loop
    best_placements = None
    best_vol = 0
    best_placed_boxes = []
    
    # 1. Generate diverse orders
    indices = list(range(n))
    
    # Order 1: Volume Descending
    order_vol = sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    # Order 2: Max Dimension Descending
    order_max = sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    # Order 3: Aspect Ratio (longest side) Descending
    order_aspect = sorted(indices, key=lambda i: max(boxes[i]) / min(boxes[i]) if min(boxes[i])>0 else 0, reverse=True)
    
    orders = [order_vol, order_max, order_aspect]
    
    # Run greedy for each order
    for order in orders:
        if time.time() > deadline:
            break
        p, v, pb = greedy_place(order, 'best_fit')
        if v > best_vol:
            best_vol = v
            best_placements = p
            best_placed_boxes = pb
    
    # 2. Random Restarts with Greedy
    # Allocate ~50% of time to random restarts
    restart_deadline = start_time + 115.0 * 0.5
    
    while time.time() < restart_deadline:
        random.shuffle(indices)
        # Mix of best_fit and first_fit for diversity? Best fit is generally better.
        p, v, pb = greedy_place(indices, 'best_fit')
        if v > best_vol:
            best_vol = v
            best_placements = p
            best_placed_boxes = pb
            
    # 3. Local Search on the best solution
    if best_placements is not None and best_placed_boxes:
        # Run local search for the remaining time
        # We might run it multiple times if time permits, or just once with many iterations
        if time.time() < deadline - 5:
            best_placements, best_vol, best_placed_boxes = local_search(best_placements, best_placed_boxes, max_iter=10000)
            
            # If local search improved it, we can try another round if time allows?
            # Or just run it again on the same solution?
            # Let's run it again to see if it converges further or if we can do a second pass.
            if time.time() < deadline - 2:
                best_placements, best_vol, best_placed_boxes = local_search(best_placements, best_placed_boxes, max_iter=10000)

    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
    return best_placements
