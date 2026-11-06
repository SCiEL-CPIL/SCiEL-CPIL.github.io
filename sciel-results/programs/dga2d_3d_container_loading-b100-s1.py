"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 120s limit, keep 10s margin for safety/overhead
    deadline = start_time + 110.0
    
    # Preprocess: store box dimensions and original indices
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, w, h, d))
        
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        # Generate unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def simulate_packing(order, rotations_choice):
        """
        Simulates packing a specific order of boxes with specific rotation indices.
        Uses a simple greedy placement (Bottom-Left-Far) for the given sequence.
        Returns (filled_volume, placements_dict)
        """
        placed = []
        placements = {}
        filled_vol = 0.0
        
        # Candidate positions for fast lookup
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for idx in order:
            orig_idx, w, h, d = box_data[idx]
            rw, rh, rd = rotations_choice[idx]
            
            # Update candidates based on current placed boxes (lazy update not fully efficient, 
            # but for 60 boxes, rebuilding the set from placed boxes is O(N^2) which is ~3600 ops per box, total 216k ops per sim. 
            # 1000 sims * 216k = 216M ops. Might be slow in Python. 
            # Optimization: Only add new faces of the newly placed box to candidates? 
            # No, because a box can be placed next to ANY previous box. 
            # However, the set of candidates grows. 
            # Let's stick to the simpler logic but optimize the inner loop.
            
            # Actually, for SA, we need speed. 
            # Let's use a simpler heuristic for the inner loop of SA: 
            # Just try a few specific spots or use the existing find_best_position logic but faster?
            # The existing find_best_position is O(N * Cx * Cy * Cz). 
            # With 60 boxes, N=60. Cx, Cy, Cz are at most 60. 60*60*60*60 = 1.3M checks per box. 60 boxes -> 78M checks per sim. Too slow for many restarts.
            
            # Revised Strategy for SA Inner Loop:
            # Use a simpler, faster placement heuristic. 
            # "Shelf" packing is very fast but less dense. 
            # "Best Fit" with limited candidate points is better.
            
            # Let's try a hybrid: 
            # 1. Sort candidates by z, x, -y.
            # 2. Check only the first K valid candidates? No, we need the best.
            
            # Alternative: Use NumPy vectorization for overlap checks?
            # Placed boxes are small (<=60). 
            # Let's optimize the Python loop.
            
            best_pos = None
            best_score = None
            
            # Generate candidates from placed boxes
            # To speed up, we can just iterate through placed boxes and consider their "corners"
            # But we also need 0.
            
            # Let's limit the number of candidate points to keep SA fast.
            # Instead of all faces, just use the faces of the last 10 placed boxes? 
            # That might miss good spots.
            
            # Let's try a different approach for the SA inner loop:
            # Simple Shelf Packing. It's O(N^2) and very fast.
            # If SA improves the ORDER, even with shelf packing, it might beat the greedy BFD.
            
            # Let's implement a fast "Shelf" packer for the SA evaluation.
            # Shelves are defined by y-levels.
            # This is a structural change in the *evaluation* function of the SA, 
            # allowing many more restarts.
            
            # Actually, let's try to make the BFD faster first.
            # If we can do ~1000-2000 simulations in 110s, we need ~50ms per sim.
            # 78M ops in 50ms is impossible in Python.
            # So we MUST use a faster heuristic for the SA inner loop.
            
            # Strategy: Use a "Greedy Shelf" heuristic for the SA inner loop.
            # It is fast (O(N^2)) and allows thousands of restarts.
            # The SA optimizes the order and rotation to make the Shelf packing better.
            
            # Let's implement Shelf Packing here.
            # State: list of shelves (y, current_x, max_h)
            # For each box, find a shelf where it fits (h <= max_h, x+w <= W).
            # If not, start a new shelf at the lowest available y.
            
            shelves = [] # List of (y, current_x, max_h)
            current_y = 0
            
            for idx in order:
                orig_idx, w, h, d = box_data[idx]
                rw, rh, rd = rotations_choice[idx]
                
                placed_on_shelf = False
                # Try to fit in existing shelves
                # Sort shelves by y (lowest first)
                # To keep it fast, just iterate.
                # If we use a simple list, insertion is O(N).
                
                best_shelf_idx = -1
                best_x = W
                
                for s_idx, (s_y, s_x, s_max_h) in enumerate(shelves):
                    if rh <= s_max_h and s_x + rw <= W:
                        # Fits in this shelf
                        if s_x < best_x:
                            best_x = s_x
                            best_shelf_idx = s_idx
                            
                if best_shelf_idx != -1:
                    s_y, s_x, s_max_h = shelves[best_shelf_idx]
                    x = s_x
                    y = s_y
                    z = 0 # Shelf packing usually assumes z=0 for simplicity or stacks in z?
                    # Standard 3D shelf packing: Shelves are horizontal (constant y). 
                    # Within a shelf, boxes are placed along x and z.
                    # This is getting complex. 
                    
                    # Let's stick to a simpler 2D-like projection or just use the existing BFD but with a TIME LIMIT per simulation.
                    # If BFD takes too long, we abort that simulation.
                    
                    pass
                
                # This is getting too complex to rewrite a fast packer on the fly.
                # Let's go back to the BFD but optimize it heavily.
                
                # Optimization 1: Pre-compute rotations for all boxes.
                # Optimization 2: Use a simpler candidate generation.
                
                # Let's try a different structural change: 
                # **Local Search on Positions**.
                # Start with a greedy BFD solution.
                # Then, try to swap rotations or move boxes to improve density.
                # This is much faster than full restarts.
                
                pass
        
        # The above thought process indicates that implementing a fast inner loop for SA is tricky.
        # Let's try a different strategy: **Iterative Improvement (Local Search)**.
        # 1. Generate an initial solution using BFD.
        # 2. Try to improve it by:
        #    a. Swapping rotations of placed boxes.
        #    b. Trying to place unplaced boxes in new positions (if space opens up).
        #    c. Swapping the order of boxes in the placement sequence (re-simulating from that point).
        
        # This is effectively a beam search or local search.
        
        # Let's implement a simple Local Search:
        # 1. BFD Initial.
        # 2. For each placed box, try all other rotations. If it still fits, keep the one that minimizes "waste" or allows better fit for others? 
        #    Hard to evaluate locally.
        
        # Let's go back to **Multi-Start BFD with Randomized Order**.
        # Instead of SA, just do 50-100 random restarts of BFD with random tie-breaking or slight randomization in order.
        # This is simpler and robust.
        
        pass

    # Let's implement the Multi-Start BFD with Randomization.
    # Strategy:
    # 1. Define a fast BFD function.
    # 2. Run it multiple times with different random perturbations to the box order (e.g., top 10% random shuffle).
    # 3. Keep the best result.

    def run_bfd(order_indices, seed_val):
        """
        Runs BFD with a specific order of box indices.
        Returns (filled_volume, placements_list)
        """
        random.seed(seed_val)
        placed = []
        placements = [None] * n # Map original index to placement
        filled_vol = 0.0
        
        # Pre-compute rotations for each box in this order
        # To speed up, we only generate rotations when needed
        
        # Candidate sets
        x_cands = {0}
        y_cands = {0}
        z_cands = {0}
        
        for idx in order_indices:
            orig_idx, w, h, d = box_data[idx]
            
            # Generate rotations
            rots = get_rotations(w, h, d)
            # Randomize rotation order to break ties differently in each run
            random.shuffle(rots)
            
            placed_this = False
            
            for rw, rh, rd in rots:
                # Filter candidates
                valid_x = [x for x in x_cands if x + rw <= W]
                valid_y = [y for y in y_cands if y + rh <= H]
                valid_z = [z for z in z_cands if z + rd <= D]
                
                if not valid_x or not valid_y or not valid_z:
                    continue
                
                best_pos = None
                best_score = None
                
                # Iterate candidates
                # To speed up, limit the number of candidates checked?
                # Or just iterate all.
                
                for z in sorted(valid_z):
                    for x in sorted(valid_x):
                        for y in sorted(valid_y):
                            # Check overlap
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in placed:
                                if not (x + rw <= px or px + pw <= x or 
                                        y + rh <= py or py + ph <= y or 
                                        z + rd <= pz or pz + pd <= z):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                score = (z, x, -y)
                                if best_score is None or score < best_score:
                                    best_score = score
                                    best_pos = (x, y, z)
                                    
                if best_pos is not None:
                    x, y, z = best_pos
                    placed.append((x, y, z, rw, rh, rd))
                    placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                    filled_vol += rw * rh * rd
                    
                    # Update candidates
                    x_cands.add(x + rw)
                    y_cands.add(y + rh)
                    z_cands.add(z + rd)
                    
                    placed_this = True
                    break
            
            # If not placed, it remains None (left out)
            
        return filled_vol, placements

    # Main Loop: Multi-Start
    best_vol = 0
    best_placements = None
    
    # Initial deterministic run (BFD by volume)
    initial_order = sorted(range(n), key=lambda i: box_data[i][1]*box_data[i][2]*box_data[i][3], reverse=True)
    
    vol, pls = run_bfd(initial_order, 42)
    if vol > best_vol:
        best_vol = vol
        best_placements = pls
    
    # Random restarts
    # How many can we do?
    # Let's estimate time per run.
    t_start = time.time()
    
    max_restarts = 50
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < max_restarts:
        # Create a perturbed order
        # Take the top 20% largest boxes and shuffle them among themselves?
        # Or just shuffle the whole order? Shuffling whole order is likely to be worse.
        # Perturb the initial sorted order.
        
        order = list(initial_order)
        
        # Randomly swap a few adjacent elements
        for _ in range(5):
            i = random.randint(0, n - 2)
            order[i], order[i+1] = order[i+1], order[i]
            
        # Also randomize rotation preference by changing seed
        seed_val = random.randint(0, 10000)
        
        vol, pls = run_bfd(order, seed_val)
        
        if vol > best_vol:
            best_vol = vol
            best_placements = pls
            
        restarts_done += 1
        
    # Finalize placements
    if best_placements:
        for i in range(n):
            if best_placements[i] is not None:
                final_placements[i] = best_placements[i]
            else:
                # Ensure it's marked as left out with original dims
                final_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    else:
        # Fallback: all left out
        for i in range(n):
            final_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return final_placements
# EVOLVE-BLOCK-END
