"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import itertools
import random

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute all 6 rotations for each box
    def get_rotations(w, h, d):
        # All 6 unique permutations of (w, h, d)
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots

    # Sort boxes: largest volume first, with tie-breaking on max dimension
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # Precompute rotations for each box in the sorted order
    box_rotations = []
    for orig_idx, (bw, bh, bd) in indexed_boxes:
        box_rotations.append(get_rotations(bw, bh, bd))

    # Placed boxes: list of (x, y, z, w, h, d)
    placed = []
    results = [None] * n  # Will store (bin, x, y, z, w, h, d) for each original box index
    
    def fits(x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        # Check overlap with all placed boxes
        for px, py, pz, pw, ph, pd in placed:
            # Overlap if intervals overlap on all 3 axes
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True
    
    def get_candidate_axes():
        xs = {0}
        ys = {0}
        zs = {0}
        for px, py, pz, pw, ph, pd in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        return sorted(xs), sorted(ys), sorted(zs)

    def pack_with_order(order_indices, rotations_choice):
        """
        Pack boxes according to the given order and rotation choices.
        Returns (filled_volume, results_list)
        """
        local_placed = []
        local_results = [None] * n
        filled_vol = 0.0
        
        # Reset candidate axes
        xs = [0]
        ys = [0]
        zs = [0]
        
        for pos_idx in order_indices:
            orig_idx, (bw, bh, bd) = indexed_boxes[pos_idx]
            rw, rh, rd = rotations_choice[pos_idx]
            
            # Get current candidate axes
            # Note: In a real implementation, we'd maintain these incrementally.
            # For simplicity in the inner loop of SA/Local Search, rebuilding is O(N^2) total.
            # With N=60, this is fine.
            xs, ys, zs = get_candidate_axes_from_list(local_placed)
            
            best_pos = None
            best_key = None
            
            for cx in xs:
                if cx + rw > W:
                    continue
                for cy in ys:
                    if cy + rh > H:
                        continue
                    for cz in zs:
                        if cz + rd > D:
                            continue
                        if not fits_in_list(cx, cy, cz, rw, rh, rd, local_placed):
                            continue
                        # BLF: minimize z, then x, then y
                        key = (cz, cx, cy)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (cx, cy, cz, rw, rh, rd)
            
            if best_pos is not None:
                cx, cy, cz, rw, rh, rd = best_pos
                local_placed.append((cx, cy, cz, rw, rh, rd))
                local_results[orig_idx] = (0, cx, cy, cz, rw, rh, rd)
                filled_vol += rw * rh * rd
            else:
                local_results[orig_idx] = (-1, 0, 0, 0, bw, bh, bd)
                
        # Fill any remaining None
        for i in range(n):
            if local_results[i] is None:
                local_results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                
        return filled_vol, local_results

    def get_candidate_axes_from_list(placed_list):
        xs = {0}
        ys = {0}
        zs = {0}
        for px, py, pz, pw, ph, pd in placed_list:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        return sorted(xs), sorted(ys), sorted(zs)

    def fits_in_list(x, y, z, bw, bh, bd, placed_list):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        for px, py, pz, pw, ph, pd in placed_list:
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True

    # 1. Initial Greedy Pass
    # Use volume-sorted order and best-fit rotation (determined by BLF placement)
    current_order = list(range(n))
    current_rotations = [box_rotations[i][0] for i in range(n)] # Default to first rotation
    
    # Perform initial packing to get a baseline
    # Note: The standard greedy picks the best rotation dynamically. 
    # For the local search to work, we need a fixed mapping of box -> rotation index.
    # Let's run a quick greedy to determine the "best" rotation for each box in the initial order.
    
    def greedy_pack():
        local_placed = []
        local_results = [None] * n
        filled_vol = 0.0
        local_rot_choices = [None] * n
        
        for pos_idx in range(n):
            orig_idx, (bw, bh, bd) = indexed_boxes[pos_idx]
            rots = box_rotations[pos_idx]
            
            xs, ys, zs = get_candidate_axes_from_list(local_placed)
            
            best_pos = None
            best_key = None
            best_rot = None
            
            for rw, rh, rd in rots:
                for cx in xs:
                    if cx + rw > W:
                        continue
                    for cy in ys:
                        if cy + rh > H:
                            continue
                        for cz in zs:
                            if cz + rd > D:
                                continue
                            if not fits_in_list(cx, cy, cz, rw, rh, rd, local_placed):
                                continue
                            key = (cz, cx, cy)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_pos = (cx, cy, cz, rw, rh, rd)
                                best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                cx, cy, cz, rw, rh, rd = best_pos
                local_placed.append((cx, cy, cz, rw, rh, rd))
                local_results[orig_idx] = (0, cx, cy, cz, rw, rh, rd)
                local_rot_choices[pos_idx] = (rw, rh, rd)
                filled_vol += rw * rh * rd
            else:
                local_results[orig_idx] = (-1, 0, 0, 0, bw, bh, bd)
                local_rot_choices[pos_idx] = rots[0] # Fallback
                
        for i in range(n):
            if local_results[i] is None:
                local_results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                
        return filled_vol, local_results, local_rot_choices

    best_vol, best_results, best_rot_choices = greedy_pack()
    best_order = list(range(n))
    
    # 2. Local Search: Simulated Annealing on Order and Rotations
    # We will swap positions in the order and swap rotations.
    
    current_vol = best_vol
    current_results = best_results
    current_rot_choices = best_rot_choices
    current_order = best_order
    
    # Map from position in order to rotation tuple
    # current_rot_choices is indexed by position in the order (0..n-1)
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    steps = 0
    max_steps = 50000 # Limit steps to ensure we finish in time
    
    # Precompute rotation indices for faster swapping? 
    # Let's just store the tuples.
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Generate neighbor
        new_order = current_order[:]
        new_rot_choices = current_rot_choices[:]
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap order of two boxes
            i, j = random.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            # Rotations stay attached to the box position? 
            # Usually, rotation is a property of the box instance. 
            # If we swap positions, the box moves, so its rotation choice moves with it.
            new_rot_choices[i], new_rot_choices[j] = new_rot_choices[j], new_rot_choices[i]
        else:
            # Change rotation of one box
            i = random.randint(0, n-1)
            # Get all possible rotations for this box
            pos_idx = new_order[i]
            orig_idx, dims = indexed_boxes[pos_idx]
            rots = box_rotations[pos_idx]
            if len(rots) > 1:
                # Pick a different rotation
                current_rot = new_rot_choices[i]
                other_rots = [r for r in rots if r != current_rot]
                if other_rots:
                    new_rot_choices[i] = random.choice(other_rots)
        
        # Evaluate neighbor
        # Optimization: We only need to re-pack from the point of change? 
        # No, changing order/rotation affects subsequent placements. 
        # Full re-pack is O(N^3) roughly. 
        # With N=60, 60^3 = 216,000 ops. 
        # 50,000 steps * 216,000 = 10^10 ops. Too slow in Python.
        
        # We need a faster evaluation or fewer steps.
        # Let's reduce max_steps significantly or optimize the packer.
        # The packer is the bottleneck.
        
        # Alternative: Use a simpler heuristic for the SA evaluation?
        # Or just do a very limited number of steps.
        
        # Let's check time. If we do 1000 steps, that's 1000 * (pack time).
        # Pack time for N=60:
        # Inner loop: sum over boxes of (num_candidates^3 * num_placed)
        # Avg candidates ~ N/2 = 30. 30^3 = 27,000.
        # Avg placed ~ N/2 = 30.
        # Per box: 27,000 * 30 = 810,000 checks.
        # Total per pack: 60 * 810,000 = 48,600,000 checks.
        # In Python, ~10^7 ops/s? No, simple loops are ~10^6-10^7.
        # So 48M checks might take 5-10 seconds.
        # That means we can only do ~10-20 SA steps in 100s.
        
        # This local search is too slow with full re-packing.
        # We need to either:
        # 1. Use a much faster packer (e.g., Shelf) for SA.
        # 2. Do very few steps (e.g., 5-10).
        # 3. Use a different improvement strategy.
        
        # Given the constraints, let's try a few "random restarts" of the greedy with random initial rotations/orders?
        # Or just a very short SA with a faster approximate packer?
        
        # Let's implement a fast "Shelf" packer for the SA evaluation.
        # Shelf packing is O(N^2).
        
        def fast_shelf_pack(order_indices, rotations_choice):
            # Shelves are horizontal layers (constant z).
            # Within a shelf, we pack in 2D (x, y).
            # This is a simplification.
            
            # State: list of shelves. Each shelf has (z, current_y, max_h)
            # Wait, standard 3D shelf:
            # Shelves are at fixed z levels.
            # Inside a shelf, we have rows (fixed y).
            # This is getting complex to implement correctly and quickly.
            
            # Let's stick to the existing BFD but limit the candidate points.
            # If we limit candidates to just the origin and the "front" faces of the last 5 boxes?
            # That might miss spots.
            
            # Let's try a different approach:
            # Since full re-pack is too slow, we will only do a VERY small number of SA steps (e.g., 20).
            # And we will use the existing packer.
            
            # Re-evaluating time:
            # If 1 pack takes 2s, 20 steps take 40s.
            # We have 100s.
            # So we can do ~30-40 steps.
            
            # Let's just run the loop with a tight time check.
            
            # We won't use the fast_shelf_pack, we will use the existing pack_with_order.
            # But we need to be careful with the time.
            
            pass # Placeholder

        # Let's just do a few iterations of "random rotation swap" and "random order swap"
        # and keep the best.
        
        # We already have best_vol, best_results.
        
        # To save time, let's just do a simple random perturbation and re-pack.
        
        # Actually, the current code structure is getting messy.
        # Let's simplify:
        # 1. Greedy Pack (done)
        # 2. Try 10 random restarts with random order/rotations?
        #    Random order is likely worse.
        #    Random rotations might help if the greedy got stuck.
        
        # Let's try: For each box, randomly pick one of its rotations, then run Greedy Pack.
        # Do this 5 times.
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Random rotations
            rand_rot_choices = [box_rotations[i][random.randint(0, len(box_rotations[i])-1)] for i in range(n)]
            
            # Run greedy pack with these fixed rotations?
            # The greedy pack currently picks the best rotation dynamically.
            # We need a version that respects fixed rotations.
            
            def pack_with_fixed_rotations(order_indices, rot_choices):
                local_placed = []
                local_results = [None] * n
                filled_vol = 0.0
                
                for pos_idx in order_indices:
                    orig_idx, (bw, bh, bd) = indexed_boxes[pos_idx]
                    rw, rh, rd = rot_choices[pos_idx]
                    
                    xs, ys, zs = get_candidate_axes_from_list(local_placed)
                    
                    best_pos = None
                    best_key = None
                    
                    for cx in xs:
                        if cx + rw > W:
                            continue
                        for cy in ys:
                            if cy + rh > H:
                                continue
                            for cz in zs:
                                if cz + rd > D:
                                    continue
                                if not fits_in_list(cx, cy, cz, rw, rh, rd, local_placed):
                                    continue
                                key = (cz, cx, cy)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_pos = (cx, cy, cz, rw, rh, rd)
                    
                    if best_pos is not None:
                        cx, cy, cz, rw, rh, rd = best_pos
                        local_placed.append((cx, cy, cz, rw, rh, rd))
                        local_results[orig_idx] = (0, cx, cy, cz, rw, rh, rd)
                        filled_vol += rw * rh * rd
                    else:
                        local_results[orig_idx] = (-1, 0, 0, 0, bw, bh, bd)
                        
                for i in range(n):
                    if local_results[i] is None:
                        local_results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                        
                return filled_vol, local_results

            vol, res = pack_with_fixed_rotations(current_order, rand_rot_choices)
            if vol > best_vol:
                best_vol = vol
                best_results = res
                # Update best_rot_choices? Not needed for final result, but good for state.
                
    # Return the best result found
    return best_results

# EVOLVE-BLOCK-END
