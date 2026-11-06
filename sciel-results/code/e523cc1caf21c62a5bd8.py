"""
Strategy: 3D Skyline / Heightmap Packing.
Instead of checking overlap against a list of placed boxes (O(N)), we maintain a 3D
grid of heights (or a 2D grid of heights per XY column, effectively a 3D skyline).
Since box dimensions are integers (implied by the grid nature of such tasks or we
can quantize to smallest dimension), we can use a heightmap.
However, to be robust to non-integer or large dimensions, we will use a
"Discrete Skyline" approach where we maintain a set of "skyline segments" in 3D.
Actually, a simpler and very effective structural change for 3D is maintaining
a 3D occupancy grid if the container is small enough, or a 2D heightmap if we
pack in layers.
Given the "60-box" constraint and typical container sizes, a 3D grid might be too
large if dimensions are large.
Let's use a **2D Skyline per Z-layer** approach? No, that's complex.

Better Structural Change: **3D Skyline with Axis-Aligned Bounding Boxes (AABB) Tree**?
No, standard library only.

Let's stick to the prompt's suggestion: "skyline or floor-map representation that
maintains a 2D grid of minimum heights for each cell".
This implies we treat the X-Y plane as a grid. For each (x,y) cell, we store the
minimum Z height where a box can be placed.
To handle arbitrary box sizes, we need to check all cells covered by the box's
footprint. If the box dimensions are integers, this is fast.
If dimensions are floats, we must quantize.
Let's assume integer dimensions or quantize to the smallest unit.
Actually, the task says "any axis rotation allowed".
Let's implement a **3D Skyline** using a list of "blocks" but optimized with a
**Best-Fit** heuristic that uses the skyline to quickly find the lowest Z.

Wait, the prompt specifically asks for: "skyline or floor-map representation that
maintains a 2D grid of minimum heights for each cell".
This is a **2.5D Skyline**. We maintain `height[x][y]`.
When placing a box at (x, y, z) with size (w, h, d), we check:
1. For all (cx, cy) in [x, x+w) x [y, y+h):
   if `height[cx][cy] > z`: overlap.
   else: `height[cx][cy] = max(height[cx][cy], z + d)`
This is O(w*h) per check.
To make it efficient, we need the grid to be small.
If the container is large, this is slow.
However, for 60 boxes, the grid might be manageable if we only update affected areas.
Let's try this. It is structurally different from list-based O(N) checks.
"""

import time
import itertools
import random

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    # Quantize to integer grid for the skyline approach.
    # If dimensions are floats, we scale them.
    # Assuming standard integer-like dimensions for grid efficiency.
    # If not, we can round to nearest integer.
    W_int = int(W)
    H_int = int(H)
    D_int = int(D)
    
    # If container is too large for a dense grid, fall back to list-based or other.
    # For 60 boxes, a grid of 100x100 is 10k cells, which is fine.
    # If W*H > 10000, we might struggle with memory/time for dense updates.
    # Let's use a sparse dictionary for the heightmap to handle large containers.
    
    deadline = time.time() + 110.0
    
    # Sparse Skyline: dict (x, y) -> min_z
    # Actually, a dict of (x,y) -> z is not enough because a box covers a range.
    # We need to know the height at every cell.
    # If we use a dict, we must check all cells in the footprint.
    
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)
    
    def try_pack_skyline(order, use_random_restarts=0):
        """
        Pack boxes using a 2D heightmap (skyline) approach.
        """
        # heightmap: dict (x, y) -> z (minimum free height)
        # Initially all 0.
        heightmap = {}
        placed = [] # list of (idx, x, y, z, w, h, d)
        
        def get_height(x, y):
            return heightmap.get((x, y), 0)
            
        def set_height(x, y, z):
            if z > get_height(x, y):
                heightmap[(x, y)] = z
                
        def check_and_place(x, y, z, bw, bh, bd, idx):
            # Check if box at (x,y,z) with dims (bw,bh,bd) fits
            # Footprint is x..x+bw-1, y..y+bh-1
            # We need to check if any cell in footprint has height > z
            # And if so, we cannot place.
            # If we place, we update height for all cells in footprint to max(current, z+bd)
            
            # Optimization: Check bounds first
            if x < 0 or y < 0 or z < 0:
                return False
            if x + bw > W_int or y + bh > H_int or z + bd > D_int:
                return False
            
            # Check overlap
            # Iterate over footprint
            # This is O(bw*bh). If bw*bh is large, this is slow.
            # For 60 boxes, average size might be small.
            for cx in range(x, x + bw):
                for cy in range(y, y + bh):
                    if get_height(cx, cy) > z:
                        return False
            
            # Place
            for cx in range(x, x + bw):
                for cy in range(y, y + bh):
                    set_height(cx, cy, z + bd)
            
            placed.append((idx, x, y, z, bw, bh, bd))
            return True
            
        # Candidate positions generation
        # We maintain a set of "candidate" (x, y) pairs.
        # Initially (0,0).
        # When a box is placed at (x, y, z, w, h, d), we add new candidates:
        # (x+w, y), (x, y+h), (x+w, y+h), etc.
        # But in 3D skyline, the "lowest Z" for a given (x,y) is determined by the heightmap.
        # So for each candidate (x,y), we compute z = max(heightmap[cx][cy] for cx in x..x+bw-1, cy in y..y+bh-1)
        # Then check if it fits in Z.
        
        candidates = [(0, 0)]
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            best_pos = None
            best_orient = None
            best_score = None
            
            # Sort candidates by "potential" (e.g., lowest Z, or closest to origin)
            # To be efficient, we can limit the number of candidates checked.
            # Let's check all candidates but break early if we find a "good" one?
            # No, Best-Fit is better.
            
            # We need to find the placement that minimizes Z, then Y, then X.
            # For each orientation, we need to find the best (x,y) from candidates.
            
            # To speed up, we can pre-filter candidates.
            # Let's just iterate through candidates.
            
            for (cx, cy) in candidates:
                for (bw, bh, bd) in orients:
                    # Calculate minimum Z for this footprint at (cx, cy)
                    min_z = 0
                    for fx in range(cx, cx + bw):
                        for fy in range(cy, cy + bh):
                            h_val = get_height(fx, fy)
                            if h_val > min_z:
                                min_z = h_val
                                
                    if min_z + bd <= D_int:
                        # Check if it actually fits (no overlap)
                        # The min_z calculation ensures that at min_z, all cells are free?
                        # No! It ensures that min_z is the *highest* of the min heights.
                        # So at z=min_z, all cells in the footprint have height <= min_z.
                        # So it fits.
                        
                        # Score: lower Z is better, then lower Y, then lower X
                        score = (min_z, cy, cx)
                        
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, min_z)
                            best_orient = (bw, bh, bd)
                            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                check_and_place(bx, by, bz, bw, bh, bd, idx)
                
                # Update candidates
                # Add neighbors of the placed box
                new_cands = [
                    (bx + bw, by),
                    (bx, by + bh),
                    (bx + bw, by + bh),
                    (bx - 1, by), # Just in case? No, we only move forward.
                    (bx, by - 1),
                ]
                # Actually, standard skyline adds (x+w, y), (x, y+h), (x+w, y+h)
                # And also (x, y) is already there.
                # We should add (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)
                # And also potentially (bx + bw, by + 1), etc?
                # The set of "corners" is sufficient for orthogonal packing.
                
                candidates.extend(new_cands)
                # Remove duplicates and filter out of bounds
                candidates = list(set([c for c in candidates if 0 <= c[0] < W_int and 0 <= c[1] < H_int]))
                
            # If no box placed, we can break or continue?
            # If we can't place this box, we leave it out.
            
        # Build result
        result = [None] * len(boxes)
        for p in placed:
            idx, x, y, z, w, h, d = p
            result[idx] = (0, x, y, z, w, h, d)
        for i in range(len(boxes)):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                
        # Calculate filled volume
        filled = 0
        for r in result:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        return result, filled

    # Define orders
    indexed_boxes = list(range(len(boxes)))
    order_vol = sorted(indexed_boxes, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    order_max = sorted(indexed_boxes, key=lambda i: max(boxes[i]), reverse=True)
    
    best_result = None
    best_volume = -1
    
    # Try different orders
    for order in [order_vol, order_max]:
        if time.time() > deadline:
            break
        res, vol = try_pack_skyline(order)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    # Randomized restarts with skyline
    # We can randomize the order slightly or the candidate selection.
    # Let's try a few random shuffles of the volume order.
    for _ in range(5):
        if time.time() > deadline:
            break
        rand_order = order_vol[:]
        random.shuffle(rand_order)
        # Or better: sort by volume with some noise
        # Let's just shuffle the non-critical part
        res, vol = try_pack_skyline(rand_order)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    if best_result is None:
        # Fallback
        placements = []
        for w, h, d in boxes:
            placements.append((-1, 0, 0, 0, w, h, d))
        return placements
        
    return best_result

# EVOLVE-BLOCK-END
