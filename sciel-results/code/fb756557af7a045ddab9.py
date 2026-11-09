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
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    class Bin:
        __slots__ = ['boxes', 'volume', 'sorted_boxes']

        def __init__(self):
            self.boxes = []  # list of (idx, x, y, z, w, h, d)
            self.volume = 0
            self.sorted_boxes = []  # sorted by x, then y, then z for pruning

        def _rebuild_sorted(self):
            self.sorted_boxes = sorted(self.boxes, key=lambda b: (b[1], b[2], b[3]))

        def can_place(self, x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Pruned overlap check: skip boxes that are clearly separated
            for (_, bx, by, bz, bw, bh, bd) in self.sorted_boxes:
                # If this box starts after our box ends in x, and all subsequent boxes do too, break
                if bx >= x + w:
                    break
                # Check if they could overlap in x
                if bx + bw <= x:
                    continue
                # Check y
                if by + bh <= y or y + h <= by:
                    continue
                # Check z
                if bz + bd <= z or z + d <= bz:
                    continue
                # They overlap
                return False
            return True

        def get_candidates(self, w, h, d):
            """Generate candidate positions using corner points from existing boxes."""
            cands = set([(0, 0, 0)])
            for (_, bx, by, bz, bw, bh, bd) in self.boxes:
                for dx in (0, bw):
                    for dy in (0, bh):
                        for dz in (0, bd):
                            cands.add((bx + dx, by + dy, bz + dz))
            
            valid_cands = []
            for (cx, cy, cz) in cands:
                if self.can_place(cx, cy, cz, w, h, d):
                    valid_cands.append((cx, cy, cz))
            
            return valid_cands

        def place(self, idx, x, y, z, w, h, d):
            self.boxes.append((idx, x, y, z, w, h, d))
            self.volume += w * h * d
            self._rebuild_sorted()

        def remove(self, idx):
            for i, box in enumerate(self.boxes):
                if box[0] == idx:
                    w, h, d = box[4], box[5], box[6]
                    self.volume -= w * h * d
                    self.boxes.pop(i)
                    self._rebuild_sorted()
                    return True
            return False

        def get_placement(self, idx):
            for box in self.boxes:
                if box[0] == idx:
                    return box[1], box[2], box[3], box[4], box[5], box[6]
            return None

        def is_empty(self):
            return len(self.boxes) == 0

    def compute_score(assignments):
        """Compute score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not assignments:
            return float('inf')
        bins_used = set(assignments.values())
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(box_vols[idx] for idx, b in assignments.items() if b == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    def greedy_pack(order, bins, assignments):
        """Place boxes in given order using best-fit heuristic."""
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
                
            placed = False
            if bins:
                # Try all existing bins, pick the one that minimizes (x+y+z) for compactness
                best_bin_id = -1
                best_pos = None
                best_metric = float('inf')
                
                for b_id in range(len(bins)):
                    cands = bins[b_id].get_candidates(w, h, d)
                    if cands:
                        # Pick candidate that minimizes (x+y+z) for compactness
                        cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
                        cx, cy, cz = cands[0]
                        metric = cx + cy + cz
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_id = b_id
                            best_pos = (cx, cy, cz)
                
                if best_bin_id >= 0:
                    cx, cy, cz = best_pos
                    bins[best_bin_id].place(idx, cx, cy, cz, w, h, d)
                    assignments[idx] = best_bin_id
                    placed = True
            
            if not placed:
                # Open new bin
                b_id = len(bins)
                bins.append(Bin())
                bins[b_id].place(idx, 0, 0, 0, w, h, d)
                assignments[idx] = b_id

    def try_move_box(assignments, bins, idx, from_bin_id, to_bin_id, w, h, d):
        """Try to move box idx from from_bin_id to to_bin_id. Returns True if moved."""
        cands = bins[to_bin_id].get_candidates(w, h, d)
        if cands:
            cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
            cx, cy, cz = cands[0]
            if bins[to_bin_id].can_place(cx, cy, cz, w, h, d):
                bins[from_bin_id].remove(idx)
                bins[to_bin_id].place(idx, cx, cy, cz, w, h, d)
                assignments[idx] = to_bin_id
                return True
        return False

    def local_search(bins, assignments):
        """Local search to minimize B, then maximize last-bin fill."""
        if time.time() > deadline:
            return

        # Phase 1: Minimize number of bins by moving boxes out of higher-numbered bins
        for _pass in range(20):
            if time.time() > deadline:
                break
            
            occupied_bins = sorted([b for b in range(len(bins)) if not bins[b].is_empty()], reverse=True)
            made_progress = False
            
            for b_id in occupied_bins:
                if time.time() > deadline:
                    break
                if bins[b_id].is_empty():
                    continue
                
                # Try to move each box in this bin to an earlier bin
                boxes_in_bin = list(bins[b_id].boxes)
                
                for (idx, x, y, z, w, h, d) in boxes_in_bin:
                    if time.time() > deadline:
                        break
                    # Verify box is still in this bin
                    if not any(b[0] == idx for b in bins[b_id].boxes):
                        continue
                    
                    # Try to move to lower-numbered bins
                    moved = False
                    for target_b in range(b_id):
                        if bins[target_b].is_empty():
                            continue
                        if try_move_box(assignments, bins, idx, b_id, target_b, w, h, d):
                            moved = True
                            made_progress = True
                            break
                    
                    # If not moved to lower bin, try to any other bin
                    if not moved:
                        for target_b in range(len(bins)):
                            if target_b == b_id or bins[target_b].is_empty():
                                continue
                            if try_move_box(assignments, bins, idx, b_id, target_b, w, h, d):
                                made_progress = True
                                break
            
            if not made_progress:
                break

        # Phase 2: Improve last bin fill by moving boxes into it
        occupied_bins = [b for b in range(len(bins)) if not bins[b].is_empty()]
        if not occupied_bins:
            return
        
        last_bin_id = max(occupied_bins)
        
        other_bins = [b for b in range(len(bins)) if b != last_bin_id and not bins[b].is_empty()]
        other_bins.sort(key=lambda b: -bins[b].volume)
        
        for b_id in other_bins:
            if time.time() > deadline:
                break
            boxes_in_bin = list(bins[b_id].boxes)
            for (idx, x, y, z, w, h, d) in boxes_in_bin:
                if not any(b[0] == idx for b in bins[b_id].boxes):
                    continue
                if try_move_box(assignments, bins, idx, b_id, last_bin_id, w, h, d):
                    pass

    # --- Main Packing Logic ---
    
    # Sort boxes by volume descending, keep original indices
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))
    
    bins = []
    assignments = {}
    
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
                    cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
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
        local_search(bins, assignments)

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
            for j in range(i + 1, len(bl)):
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
