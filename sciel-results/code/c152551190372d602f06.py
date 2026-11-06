"""3D Container Loading with BSP tree + multi-strategy + local search polish."""
# EVOLVE-BLOCK-START
import time
import math
import random
import heapq
from collections import defaultdict

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    # Preprocess
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, float(w), float(h), float(d)))
    
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        """Get all unique rotations, sorted by height then width."""
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return sorted(list(rots), key=lambda r: (r[1], r[0], r[2]))
    
    def can_place(placed, x, y, z, bw, bh, bd):
        """Check if box can be placed at (x,y,z) without overlap."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True
    
    # ============ BSP TREE APPROACH ============
    
    class BSPNode:
        """Binary Space Partitioning node representing a rectangular region."""
        __slots__ = ['x', 'y', 'z', 'w', 'h', 'd', 'left', 'right', 'occupied']
        
        def __init__(self, x, y, z, w, h, d):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
            self.h = h
            self.d = d
            self.left = None
            self.right = None
            self.occupied = False  # True if a box is placed here
    
    def bsp_place(root, bw, bh, bd):
        """
        Try to place a box of dimensions (bw, bh, bd) into the BSP tree.
        Returns (x, y, z) if successful, None otherwise.
        Uses best-fit: find the smallest free region that fits.
        """
        if root is None or root.occupied:
            return None
        
        # If this node fits the box and is empty, we can potentially place here
        # But we need to check if we should place here or in a child
        
        if root.left is None and root.right is None:
            # Leaf node
            if bw <= root.w and bh <= root.h and bd <= root.d:
                # Place box at the corner of this region
                x, y, z = root.x, root.y, root.z
                # Partition: mark as occupied and create children for remaining space
                _partition_around_box(root, x, y, z, bw, bh, bd)
                return (x, y, z)
            return None
        
        # Try left child first (bottom), then right
        result = bsp_place(root.left, bw, bh, bd)
        if result is not None:
            return result
        result = bsp_place(root.right, bw, bh, bd)
        if result is not None:
            return result
        
        # Neither child worked, but maybe this node itself can accommodate
        # This happens if the box spans both children
        if bw <= root.w and bh <= root.h and bd <= root.d:
            x, y, z = root.x, root.y, root.z
            _partition_around_box(root, x, y, z, bw, bh, bd)
            return (x, y, z)
        
        return None
    
    def _partition_around_box(node, x, y, z, bw, bh, bd):
        """
        After placing a box at (x,y,z) with dims (bw,bh,bd) in node,
        partition the remaining space into free regions.
        Simplified: create up to 6 sub-regions around the box.
        We use a simplified approach: split into left/right based on x.
        """
        node.occupied = True
        
        # The box occupies [x, x+bw] x [y, y+bh] x [z, z+bd]
        # Remaining space in node:
        # Left: [node.x, x] if x > node.x
        # Right: [x+bw, node.x+node.w] if x+bw < node.x+node.w
        # Similarly for y and z
        
        # For simplicity, we'll use a "skyline" like approach within the BSP:
        # Just create two children: left part and right part (split along x-axis)
        
        remaining_w = node.w
        box_offset_x = x - node.x
        
        # Left region
        if box_offset_x > 0:
            left_w = box_offset_x
            node.left = BSPNode(node.x, node.y, node.z, left_w, node.h, node.d)
        
        # Right region
        right_w = remaining_w - box_offset_x - bw
        if right_w > 0:
            node.right = BSPNode(x + bw, node.y, node.z, right_w, node.h, node.d)
        
        # Also need to handle y and z splits - this is simplified
        # A more complete BSP would create up to 8 sub-regions
    
    def bsp_packing(boxes_list, deadline):
        """Packing strategy using BSP tree."""
        root = BSPNode(0, 0, 0, W, H, D)
        placements = {}
        placed = []
        
        # Sort by volume descending
        sorted_boxes = sorted(boxes_list, key=lambda b: b[1]*b[2]*b[3], reverse=True)
        
        for idx, w, h, d in sorted_boxes:
            if time.time() > deadline:
                break
            
            rots = get_rotations(w, h, d)
            placed_this = False
            
            for rw, rh, rd in rots:
                pos = bsp_place(root, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, rw, rh, rd))
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_this = True
                    break
        
        return placed, placements
    
    # ============ CANDIDATE POSITION APPROACH (improved) ============
    
    def find_best_fit(placed, bw, bh, bd):
        """Find best position using candidate points from placed boxes."""
        if bw > W or bh > H or bd > D:
            return None
        
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
        
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None
        
        best_score = None
        best_coord = None
        
        # Sort candidates to prefer lower z, then lower x, then lower y
        for z in sorted(z_candidates):
            for x in sorted(x_candidates):
                for y in sorted(y_candidates):
                    if can_place(placed, x, y, z, bw, bh, bd):
                        score = (z, x, y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            return best_coord
                            
        return None
    
    def multi_pass_packing(boxes_list, deadline):
        """Multi-pass best-fit decreasing packing."""
        placed = []
        placements = {}
        unplaced = list(boxes_list)
        
        max_passes = 10
        for p in range(max_passes):
            if time.time() > deadline:
                break
            
            if not unplaced:
                break
            
            # Sort by volume descending
            unplaced.sort(key=lambda b: b[1]*b[2]*b[3], reverse=True)
            
            newly_placed = 0
            still_unplaced = []
            
            for idx, w, h, d in unplaced:
                if time.time() > deadline:
                    still_unplaced.append((idx, w, h, d))
                    continue
                
                rots = get_rotations(w, h, d)
                placed_this = False
                
                for rw, rh, rd in rots:
                    pos = find_best_fit(placed, rw, rh, rd)
                    if pos is not None:
                        x, y, z = pos
                        placed.append((x, y, z, rw, rh, rd))
                        placements[idx] = (0, x, y, z, rw, rh, rd)
                        placed_this = True
                        newly_placed += 1
                        break
                        
                if not placed_this:
                    still_unplaced.append((idx, w, h, d))
                    
            if newly_placed == 0:
                break
            unplaced = still_unplaced
        
        return placed, placements
    
    def multi_sort_packing(boxes_list, deadline):
        """Try multiple sorting strategies."""
        best_placed = []
        best_placements = {}
        best_vol = 0
        
        sort_keys = [
            lambda b: b[1]*b[2]*b[3],           # volume
            lambda b: max(b[1], b[2], b[3]),    # max dimension
            lambda b: b[1]*b[2] + b[1]*b[3] + b[2]*b[3],  # surface area
            lambda b: b[1]*b[2]*b[3] / max(b[1], b[2], b[3], 0.001),  # volume/maxdim
        ]
        
        for sort_key in sort_keys:
            if time.time() > deadline:
                break
            
            placed = []
            placements = {}
            unplaced = list(boxes_list)
            
            unplaced.sort(key=sort_key, reverse=True)
            
            newly_placed = 0
            still_unplaced = []
            
            for idx, w, h, d in unplaced:
                if time.time() > deadline:
                    still_unplaced.append((idx, w, h, d))
                    continue
                
                rots = get_rotations(w, h, d)
                placed_this = False
                
                for rw, rh, rd in rots:
                    pos = find_best_fit(placed, rw, rh, rd)
                    if pos is not None:
                        x, y, z = pos
                        placed.append((x, y, z, rw, rh, rd))
                        placements[idx] = (0, x, y, z, rw, rh, rd)
                        placed_this = True
                        newly_placed += 1
                        break
                        
                if not placed_this:
                    still_unplaced.append((idx, w, h, d))
            
            # Second pass on unplaced
            if still_unplaced and time.time() < deadline:
                still_unplaced.sort(key=sort_key, reverse=True)
                for idx, w, h, d in still_unplaced:
                    if time.time() > deadline:
                        break
                    rots = get_rotations(w, h, d)
                    for rw, rh, rd in rots:
                        pos = find_best_fit(placed, rw, rh, rd)
                        if pos is not None:
                            x, y, z = pos
                            placed.append((x, y, z, rw, rh, rd))
                            placements[idx] = (0, x, y, z, rw, rh, rd)
                            break
            
            vol = sum(rw*rh*rd for (x,y,z,rw,rh,rd) in placed)
            if vol > best_vol:
                best_vol = vol
                best_placed = placed
                best_placements = placements
        
        return best_placed, best_placements
    
    # ============ LOCAL SEARCH POLISH ============
    
    def local_search_polish(placed, placements, boxes_list, deadline):
        """Improve packing by moving boxes to better positions."""
        current_placed = list(placed)
        current_placements = dict(placements)
        
        def compute_obj(placed_list):
            """Objective: minimize sum of z-coordinates (compactness)."""
            return sum(p[2] for p in placed_list)
        
        best_obj = compute_obj(current_placed)
        best_placed = list(current_placed)
        best_placements = dict(current_placements)
        
        T = 200.0
        cooling_rate = 0.995
        min_T = 5.0
        
        iteration = 0
        while T > min_T and time.time() < deadline:
            iteration += 1
            
            placed_idxs = [i for i, v in current_placements.items() if v[0] == 0]
            if not placed_idxs:
                break
            
            idx = random.choice(placed_idxs)
            bin_, x, y, z, rw, rh, rd = current_placements[idx]
            
            box_w, box_h, box_d = boxes[idx]
            rots = get_rotations(box_w, box_h, box_d)
            
            # Try a different rotation or position
            new_rot = random.choice(rots)
            new_rw, new_rh, new_rd = new_rot
            
            # Remove from placed
            new_placed = [p for p in current_placed if not (p[0] == x and p[1] == y and p[2] == z and 
                                                           p[3] == rw and p[4] == rh and p[5] == rd)]
            
            pos = find_best_fit(new_placed, new_rw, new_rh, new_rd)
            if pos is not None:
                new_x, new_y, new_z = pos
                candidate_placed = new_placed + [(new_x, new_y, new_z, new_rw, new_rh, new_rd)]
                
                old_obj = compute_obj(current_placed)
                new_obj = compute_obj(candidate_placed)
                
                delta = new_obj - old_obj
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.1)):
                    current_placed = candidate_placed
                    current_placements[idx] = (0, new_x, new_y, new_z, new_rw, new_rh, new_rd)
                    
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_placed = list(current_placed)
                        best_placements = dict(current_placements)
            
            T *= cooling_rate
        
        return best_placed, best_placements
    
    # ============ MAIN: TRY MULTIPLE STRATEGIES ============
    
    best_result = None
    best_volume = 0.0
    
    def evaluate_result(placed, placements):
        vol = 0.0
        for idx, (bin_, x, y, z, rw, rh, rd) in placements.items():
            if bin_ == 0:
                vol += rw * rh * rd
        return vol
    
    # Strategy 1: Multi-pass best-fit
    t1 = time.time()
    if t1 < deadline:
        placed1, placements1 = multi_pass_packing(box_data, min(deadline, start_time + 45))
        vol1 = evaluate_result(placed1, placements1)
        if vol1 > best_volume:
            best_volume = vol1
            best_result = placements1
    
    # Strategy 2: Multi-sort packing
    t2 = time.time()
    if t2 < deadline:
        placed2, placements2 = multi_sort_packing(box_data, min(deadline, start_time + 75))
        vol2 = evaluate_result(placed2, placements2)
        if vol2 > best_volume:
            best_volume = vol2
            best_result = placements2
    
    # Strategy 3: BSP packing
    t3 = time.time()
    if t3 < deadline:
        placed3, placements3 = bsp_packing(box_data, min(deadline, start_time + 90))
        vol3 = evaluate_result(placed3, placements3)
        if vol3 > best_volume:
            best_volume = vol3
            best_result = placements3
    
    # Strategy 4: Local search polish on the best result
    t4 = time.time()
    if t4 < deadline and best_result:
        placed_sa = []
        for idx, (bin_, x, y, z, rw, rh, rd) in best_result.items():
            if bin_ == 0:
                placed_sa.append((x, y, z, rw, rh, rd))
        
        polish_deadline = min(deadline, start_time + 112)
        placed_polished, placements_polished = local_search_polish(
            placed_sa, best_result, box_data, polish_deadline
        )
        
        vol_polished = evaluate_result(placed_polished, placements_polished)
        if vol_polished > best_volume:
            best_volume = vol_polished
            best_result = placements_polished
    
    # Build final output
    if best_result:
        for idx in range(n):
            if idx in best_result:
                final_placements[idx] = best_result[idx]
            else:
                final_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    else:
        # Fallback: try simple greedy
        for idx, w, h, d in box_data:
            rots = get_rotations(w, h, d)
            for rw, rh, rd in rots:
                pos = find_best_fit(
                    [(x, y, z, rw2, rh2, rd2) for (x, y, z, rw2, rh2, rd2) in 
                     [(fp[1], fp[2], fp[3], fp[4], fp[5], fp[6]) for fp in final_placements if fp[0] == 0]],
                    rw, rh, rd
                )
                if pos is not None:
                    x, y, z = pos
                    final_placements[idx] = (0, x, y, z, rw, rh, rd)
                    break
    
    return final_placements
# EVOLVE-BLOCK-END
