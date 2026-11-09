"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Sort boxes by volume descending, keep original indices
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    class Bin:
        def __init__(self):
            self.boxes = []  # list of (idx, x, y, z, w, h, d)
            self.volume = 0
            # Spatial index for faster overlap checks
            self.x_intervals = []  # list of (x, x+w, box_index)
            self.y_intervals = []  # list of (y, y+h, box_index)
            self.z_intervals = []  # list of (z, z+d, box_index)

        def can_place(self, x, y, z, w, h, d, ignore_idx=None):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Quick check using interval trees
            for (x1, x2, bi) in self.x_intervals:
                if ignore_idx is not None and bi == ignore_idx:
                    continue
                if x < x2 and x + w > x1:  # x overlap
                    # Check y and z for this box
                    idx, bx, by, bz, bw, bh, bd = self.boxes[bi]
                    if y < by + bh and y + h > by and z < bz + bd and z + d > bz:
                        return False
            return True

        def get_candidates(self, w, h, d):
            """Generate candidate positions using corner points from existing boxes."""
            cands = [(0, 0, 0)]
            for (_, bx, by, bz, bw, bh, bd) in self.boxes:
                # Generate all 8 corner combinations
                for dx in [0, bw]:
                    for dy in [0, bh]:
                        for dz in [0, bd]:
                            cands.append((bx + dx, by + dy, bz + dz))
            
            seen = set()
            valid_cands = []
            for (cx, cy, cz) in cands:
                key = (cx, cy, cz)
                if key in seen:
                    continue
                seen.add(key)
                if self.can_place(cx, cy, cz, w, h, d):
                    valid_cands.append(key)
            
            # If no corner candidates fit, try (0,0,0)
            if not valid_cands:
                if self.can_place(0, 0, 0, w, h, d):
                    valid_cands = [(0, 0, 0)]
            
            return valid_cands

        def place(self, idx, x, y, z, w, h, d):
            box_idx = len(self.boxes)
            self.boxes.append((idx, x, y, z, w, h, d))
            self.volume += w * h * d
            self.x_intervals.append((x, x + w, box_idx))
            self.y_intervals.append((y, y + h, box_idx))
            self.z_intervals.append((z, z + d, box_idx))

        def remove(self, idx):
            for i, box in enumerate(self.boxes):
                if box[0] == idx:
                    w, h, d = box[4], box[5], box[6]
                    self.volume -= w * h * d
                    self.boxes.pop(i)
                    # Rebuild interval indices (simple approach for now)
                    self.x_intervals = [(b[1], b[1]+b[4], i) for i, b in enumerate(self.boxes)]
                    self.y_intervals = [(b[2], b[2]+b[5], i) for i, b in enumerate(self.boxes)]
                    self.z_intervals = [(b[3], b[3]+b[6], i) for i, b in enumerate(self.boxes)]
                    return
            # If not found, do nothing

        def get_placement(self, idx):
            for box in self.boxes:
                if box[0] == idx:
                    return box[1], box[2], box[3], box[4], box[5], box[6]
            return None

    def local_search(bins, assignments, boxes_map):
        """
        Perform local search to minimize bin count and improve last bin fill.
        """
        # Multiple passes for better convergence
        max_passes = 5
        for _pass in range(max_passes):
            if time.time() > deadline:
                break
            
            # 1. Minimize bin count by moving boxes from highest-numbered bins to others
            occupied_bins = sorted([b for b in range(len(bins)) if bins[b].boxes], reverse=True)
            
            for b_id in occupied_bins:
                if time.time() > deadline:
                    break
                if not bins[b_id].boxes:
                    continue
                
                boxes_to_try = list(bins[b_id].boxes)
                
                for (idx, x, y, z, w, h, d) in boxes_to_try:
                    if not any(b[0] == idx for b in bins[b_id].boxes):
                        continue
                    
                    moved = False
                    other_bins = [b for b in range(len(bins)) if b != b_id and bins[b].boxes]
                    other_bins.sort(key=lambda b: -bins[b].volume)
                    
                    for target_b in other_bins:
                        cands = bins[target_b].get_candidates(w, h, d)
                        if cands:
                            cands.sort(key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2]))
                            cx, cy, cz = cands[0]
                            if bins[target_b].can_place(cx, cy, cz, w, h, d):
                                bins[b_id].remove(idx)
                                bins[target_b].place(idx, cx, cy, cz, w, h, d)
                                assignments[idx] = target_b
                                moved = True
                                break

        # 2. Improve last bin fill by moving boxes into it
        occupied_bins = [b for b in range(len(bins)) if bins[b].boxes]
        if not occupied_bins:
            return
        
        last_bin_id = max(occupied_bins)
        last_bin = bins[last_bin_id]
        
        other_bins = [b for b in range(len(bins)) if b != last_bin_id and bins[b].boxes]
        other_bins.sort(key=lambda b: -bins[b].volume)
        
        for b_id in other_bins:
            if time.time() > deadline:
                break
            boxes_to_try = list(bins[b_id].boxes)
            for (idx, x, y, z, w, h, d) in boxes_to_try:
                if not any(b[0] == idx for b in bins[b_id].boxes):
                    continue
                cands = last_bin.get_candidates(w, h, d)
                if cands:
                    cands.sort(key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2]))
                    cx, cy, cz = cands[0]
                    if last_bin.can_place(cx, cy, cz, w, h, d):
                        bins[b_id].remove(idx)
                        last_bin.place(idx, cx, cy, cz, w, h, d)
                        assignments[idx] = last_bin_id

    # --- Main Packing Logic ---
    
    bins = []
    assignments = {}
    boxes_map = {idx: boxes[idx] for idx in range(n)}
    
    # Place boxes in sorted order using best-fit decreasing
    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        placed = False
        if bins:
            # Try bins in order of decreasing volume (best-fit)
            bin_indices = sorted(range(len(bins)), key=lambda i: -bins[i].volume)
            for b_id in bin_indices:
                cands = bins[b_id].get_candidates(w, h, d)
                if cands:
                    # Pick most compact position
                    cands.sort(key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2]))
                    cx, cy, cz = cands[0]
                    bins[b_id].place(idx, cx, cy, cz, w, h, d)
                    assignments[idx] = b_id
                    placed = True
                    break
        
        if not placed:
            b_id = len(bins)
            bins.append(Bin())
            bins[b_id].place(idx, 0, 0, 0, w, h, d)
            assignments[idx] = b_id

    # Run Local Search
    if time.time() < deadline:
        local_search(bins, assignments, boxes_map)

    # Construct result
    result = [None] * n
    for idx in range(n):
        if idx in assignments:
            b_id = assignments[idx]
            pos = bins[b_id].get_placement(idx)
            if pos:
                x, y, z, w, h, d = pos
                result[idx] = (b_id, x, y, z, w, h, d)
            else:
                result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        else:
            result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])

    # Verify validity
    valid = True
    bins_boxes = {}
    for idx in range(n):
        b, x, y, z, w, h, d = result[idx]
        if b not in bins_boxes:
            bins_boxes[b] = []
        bins_boxes[b].append((x, y, z, w, h, d))
        
    for b, bl in bins_boxes.items():
        for i in range(len(bl)):
            for j in range(i+1, len(bl)):
                if is_overlap(bl[i][0], bl[i][1], bl[i][2], bl[i][3], bl[i][4], bl[i][5],
                              bl[j][0], bl[j][1], bl[j][2], bl[j][3], bl[j][4], bl[j][5]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to simple shelf pack
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for idx, (w, h, d) in indexed_boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
        
        shelf_final = [None] * n
        for i, (idx, (w, h, d)) in enumerate(indexed_boxes):
            b, x, y, z, bw, bh, bd = placements[i]
            shelf_final[idx] = (b, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        return shelf_final

    return result
# EVOLVE-BLOCK-END
