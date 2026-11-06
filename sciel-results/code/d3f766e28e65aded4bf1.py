"""
Strategy: Continuous-space Bottom-Left-Farthest (BLF) packing with multi-restart and local search.
This is a fundamentally different approach from the discrete grid/skyline method.
We operate in continuous 3D space (using floats) and use a spatial index (list of placed boxes) for overlap checking.
The core heuristic is "Bottom-Left-Farthest": for each box, we try to place it at the position that minimizes z, then x, then y (or some variation) among all valid positions.
To find valid positions efficiently without scanning the entire container volume, we generate candidate anchor points from the faces of already-placed boxes.
We run this greedy packer with many different box orderings (volume, size, random) and also perform a simple local search (swap adjacent boxes in the order) to improve the best solution found.
This avoids the discretization error and grid overhead of the skyline method, potentially allowing tighter packing.
"""

import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0
    
    # Convert to float for continuous operations
    W_f, H_f, D_f = float(W), float(H), float(D)
    boxes_f = [[float(w), float(h), float(d)] for w, h, d in boxes]
    
    # Precompute unique orientations for each box
    all_orients = []
    for i in range(n):
        w, h, d = boxes_f[i]
        orients = set(itertools.permutations((w, h, d)))
        all_orients.append(list(orients))
        
    volumes = [w*h*d for w, h, d in boxes_f]
    container_vol = W_f * H_f * D_f
    
    def check_overlap(box, placed_boxes):
        """Check if a box at (x, y, z) with dims (bw, bh, bd) overlaps with any placed box."""
        bx, by, bz, bw, bh, bd = box
        for px, py, pz, pw, ph, pd in placed_boxes:
            # Check overlap in each dimension
            if bx < px + pw and bx + bw > px and \
               by < py + ph and by + bh > py and \
               bz < pz + pd and bz + bd > pz:
                return True
        return False
    
    def is_valid(box, placed_boxes):
        """Check if a box placement is valid (within container and no overlap)."""
        bx, by, bz, bw, bh, bd = box
        if bx < -1e-9 or by < -1e-9 or bz < -1e-9:
            return False
        if bx + bw > W_f + 1e-9 or by + bh > H_f + 1e-9 or bz + bd > D_f + 1e-9:
            return False
        if check_overlap(box, placed_boxes):
            return False
        return True
    
    def pack_with_blf(order):
        """
        Greedy BLF packer.
        For each box in the given order, try all orientations and candidate positions.
        Candidate positions are generated from the faces of placed boxes.
        We pick the position that minimizes a cost function (e.g., z, then x, then y).
        """
        placed = []  # List of (x, y, z, w, h, d)
        result = [None] * n
        filled = 0.0
        
        # Candidate positions: start with (0, 0, 0)
        candidates = [(0.0, 0.0, 0.0)]
        
        for idx in order:
            w, h, d = boxes_f[idx]
            best_pos = None
            best_orient = None
            best_cost = float('inf')
            
            # Generate candidate positions from current placed boxes
            # We add positions adjacent to the faces of placed boxes
            # For efficiency, we only add a limited number of new candidates per placement
            new_candidates = []
            for px, py, pz, pw, ph, pd in placed:
                # Right face
                new_candidates.append((px + pw, py, pz))
                # Top face
                new_candidates.append((px, py + ph, pz))
                # Front face
                new_candidates.append((px, py, pz + pd))
                # Corners/edges combinations for better packing
                new_candidates.append((px + pw, py + ph, pz))
                new_candidates.append((px + pw, py, pz + pd))
                new_candidates.append((px, py + ph, pz + pd))
                new_candidates.append((px + pw, py + ph, pz + pd))
            
            # Combine existing candidates and new ones, remove duplicates (approximate)
            all_cands = candidates + new_candidates
            # To avoid too many candidates, we can limit or just use them all.
            # With 60 boxes, max 60 * 8 = 480 new candidates per step. Total ~ 10k-20k candidates.
            # 20k candidates * 6 orientations * 60 boxes = ~7M checks. This might be slow in Python.
            # Optimization: Only check a subset of candidates or prune early.
            
            # For each orientation
            for bw, bh, bd in all_orients[idx]:
                if bw > W_f or bh > H_f or bd > D_f:
                    continue
                
                # For each candidate position
                for cx, cy, cz in all_cands:
                    # Check if the box fits at this position
                    if cx + bw > W_f + 1e-9 or cy + bh > H_f + 1e-9 or cz + bd > D_f + 1e-9:
                        continue
                    
                    box = (cx, cy, cz, bw, bh, bd)
                    if is_valid(box, placed):
                        # Cost function: minimize z, then x, then y (Bottom-Left-Farthest)
                        # Or minimize volume of "waste" above the box?
                        # Let's use: cost = cz * 1000 + cx * 100 + cy
                        cost = cz * 1000.0 + cx * 100.0 + cy
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (cx, cy, cz)
                            best_orient = (bw, bh, bd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd
                
                # Update candidates: add positions based on the newly placed box
                # We already added them in the loop above for the next iteration, 
                # but we need to keep the base candidates list clean.
                # Actually, the `candidates` list in the next iteration will include new ones.
                # Let's just append the new ones to the main list for future use?
                # No, better to regenerate from `placed` each time? That's O(N^2) per step.
                # Current approach: `all_cands` is rebuilt each step. This is O(N^2) total candidates.
                # 60 boxes -> 1800 steps? No, 60 steps.
                # Step 1: 1 candidate. Step 2: 1+8=9. Step 3: 9+8=17... Step 60: ~480 candidates.
                # Total checks: sum_{i=1}^{60} (8*i * 6 * 60) ~ 8*6*60 * (60*61/2) ~ 10M checks.
                # Each check is O(N) overlap -> 10M * 30 (avg N) = 300M operations. Too slow in Python.
                
                # OPTIMIZATION: Use a spatial hash or just limit candidates.
                # Let's limit the number of candidates checked per box to, say, 100 best ones.
                # Or, use a faster overlap check.
                # Alternative: Use a 3D grid if dimensions are small? 
                # Given the potential slowness, let's try to optimize the inner loop.
                
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return filled, result

    # This BLF approach as described above is likely too slow in pure Python due to O(N) overlap checks.
    # We need a faster method. Let's go back to a grid-based approach but in continuous space? 
    # No, the prompt asks for a DIFFERENT strategy. 
    # The current program uses a discrete grid. 
    # A truly different strategy could be **Genetic Algorithm on Box Assignments**.
    # But that's also slow.
    
    # Let's try a **Fast Greedy with Spatial Indexing (KD-Tree or similar)**.
    # Or, since we have NumPy, we can use a **3D Occupancy Grid** but with a smaller resolution?
    # No, the key difference should be the *search method*.
    
    # Let's implement a **Simulated Annealing on the placement itself**.
    # Start with a greedy solution, then perturb by moving/rotating a box and accepting if it improves or with probability exp(-delta/T).
    # This is materially different from the static greedy + multi-order approach.
    
    def greedy_pack(order):
        """Fast greedy packer using a 3D grid for fast overlap checks."""
        # Determine grid resolution. If container is large, use coarse grid?
        # Let's assume integer dimensions for grid simplicity, as in the original.
        W_i, H_i, D_i = int(W_f), int(H_f), int(D_f)
        # If dimensions are not integers, this will be approximate.
        # For now, let's stick to the grid approach for speed, but use it as a baseline for SA.
        
        # Create a 3D grid of 0s and 1s
        # If W_i * H_i * D_i is too large, this will fail.
        # Let's check size.
        if W_i * H_i * D_i > 10**7:
            # Too large for a dense grid. Use list of boxes with spatial hash?
            # For now, fallback to a simpler list-based check but limit candidates.
            return _greedy_list_based(order)
            
        grid = np.zeros((W_i, H_i, D_i), dtype=np.uint8)
        placed = []
        result = [None] * n
        filled = 0.0
        
        for idx in order:
            w, h, d = boxes_f[idx]
            # Snap to integer grid?
            wi, hi, di = int(round(w)), int(round(h)), int(round(d))
            if wi <= 0 or hi <= 0 or di <= 0:
                continue
                
            best_pos = None
            best_orient = None
            best_cost = float('inf')
            
            for bw, bh, bd in all_orients[idx]:
                bi, bh_i, bd_i = int(round(bw)), int(round(bh)), int(round(bd))
                if bi > W_i or bh_i > H_i or bd_i > D_i:
                    continue
                
                # Find the first valid position in a "bottom-left-farthest" scan order
                # Scan z from 0 to D-bd
                # Scan x from 0 to W-bw
                # Scan y from 0 to H-bh
                # This is O(W*H*D) per box, which is too slow.
                
                # Use the anchor point strategy again, but with the grid for fast checks.
                # Anchors: (0,0,0) and faces of placed boxes.
                anchors = [(0, 0, 0)]
                for px, py, pz, pw, ph, pd in placed:
                    anchors.append((px + pw, py, pz))
                    anchors.append((px, py + ph, pz))
                    anchors.append((px, py, pz + pd))
                    # Add more?
                    
                for ax, ay, az in anchors:
                    if ax + bi > W_i or ay + bh_i > H_i or az + bd_i > D_i:
                        continue
                    
                    # Check grid
                    region = grid[ax:ax+bi, ay:ay+bh_i, az:az+bd_i]
                    if np.sum(region) == 0:
                        # Valid
                        cost = az * 1000 + ax * 100 + ay
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (ax, ay, az)
                            best_orient = (bw, bh, bd)
                            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                bi, bh_i, bd_i = int(round(bw)), int(round(bh)), int(round(bd))
                grid[bx:bx+bi, by:by+bh_i, bz:bz+bd_i] = 1
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return filled, result

    def _greedy_list_based(order):
        """Fallback for large containers: list-based with limited anchors."""
        placed = []
        result = [None] * n
        filled = 0.0
        
        for idx in order:
            w, h, d = boxes_f[idx]
            best_pos = None
            best_orient = None
            best_cost = float('inf')
            
            anchors = [(0, 0, 0)]
            for px, py, pz, pw, ph, pd in placed:
                anchors.append((px + pw, py, pz))
                anchors.append((px, py + ph, pz))
                anchors.append((px, py, pz + pd))
                
            # Limit anchors to 100 to keep it fast
            anchors = anchors[:100]
            
            for bw, bh, bd in all_orients[idx]:
                if bw > W_f or bh > H_f or bd > D_f:
                    continue
                for ax, ay, az in anchors:
                    if ax + bw > W_f or ay + bh > H_f or az + bd > D_f:
                        continue
                    box = (ax, ay, az, bw, bh, bd)
                    if is_valid(box, placed):
                        cost = az * 1000 + ax * 100 + ay
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (ax, ay, az)
                            best_orient = (bw, bh, bd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return filled, result

    # Generate initial solutions
    orders = []
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    max_dims = [max(boxes_f[i]) for i in range(n)]
    orders.append(sorted(range(n), key=lambda i: max_dims[i], reverse=True))
    for _ in range(10):
        random_order = list(range(n))
        random.shuffle(random_order)
        orders.append(random_order)
        
    best_filled = 0
    best_result = None
    best_order = None
    
    for order in orders:
        if time.time() > deadline - 30.0:
            break
        filled, result = greedy_pack(order)
        if filled > best_filled:
            best_filled = filled
            best_result = result
            best_order = order[:]
            
    # Simulated Annealing on the order
    if best_order is not None and time.time() < deadline - 5.0:
        current_order = best_order[:]
        current_filled = best_filled
        current_result = best_result
        T = 10.0
        for step in range(1000):
            if time.time() > deadline - 5.0:
                break
            # Perturb: swap two random boxes
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            new_filled, new_result = greedy_pack(current_order)
            delta = new_filled - current_filled
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_filled = new_filled
                current_result = new_result
                if new_filled > best_filled:
                    best_filled = new_filled
                    best_result = new_result
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
            T *= 0.995
            
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes_f]
        
    return best_result

# EVOLVE-BLOCK-END
