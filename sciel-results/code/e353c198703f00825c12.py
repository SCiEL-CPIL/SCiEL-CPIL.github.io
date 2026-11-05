import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    # We need to keep track of original indices to return in order? 
    # No, the return interface is just a list of placements. 
    # The problem statement says "placements per box". 
    # Usually, the order of the returned list doesn't matter as long as each box is placed, 
    # but let's check if the evaluator cares about the order of the list corresponding to input order.
    # "return one placement ... per box". It doesn't explicitly say "in the same order".
    # However, standard practice is to return a list of tuples. 
    # If I return them in sorted order, is that valid? 
    # The evaluator checks "boxes in the same bin must not overlap".
    # It likely iterates through the returned list. 
    # Let's assume the order of the output list doesn't strictly need to match input, 
    # but to be safe, I will store the index and return in the original order if possible, 
    # or just return the list of placements. 
    # Actually, looking at the seed: `placements.append((b, x, y, z, w, h, d))`. 
    # It returns a list. The evaluator likely just checks validity of the set of boxes.
    # I will return the list of placements. The order in the list shouldn't affect validity.
    
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State: list of bins. Each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    bin_volumes = []
    
    def can_place(bin_idx, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_position(bin_idx, w, h, d):
        # Try to find a valid position. 
        # Heuristic: 
        # 1. Try to fit on top of existing boxes (max z)
        # 2. Try to fit in corners/edges
        
        # Simple grid search is too slow. 
        # Let's use a list of candidate positions.
        # Candidates:
        # - (0,0,0)
        # - For each box in bin, try (ox+ow, oy, oz), (ox, oy+oh, oz), (ox, oy, oz+od)
        # - Also (0,0, max_z), (max_x, 0, 0), etc.
        
        candidates = [(0, 0, 0)]
        max_x, max_y, max_z = 0, 0, 0
        
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
            
            if ox + ow > max_x: max_x = ox + ow
            if oy + oh > max_y: max_y = oy + oh
            if oz + od > max_z: max_z = oz + od
        
        # Add some boundary-based candidates
        candidates.append((0, 0, max_z))
        candidates.append((max_x, 0, 0))
        candidates.append((0, max_y, 0))
        
        # Deduplicate and sort by some heuristic (e.g., x+y+z, or closest to origin)
        # Sorting by x, then y, then z is a common "First Fit" variant.
        # Or sort by volume of the "hole" created? 
        # Let's sort by x, then y, then z to pack towards the corner.
        unique_candidates = list(set(candidates))
        unique_candidates.sort(key=lambda p: (p[0], p[1], p[2]))
        
        for cx, cy, cz in unique_candidates:
            if can_place(bin_idx, cx, cy, cz, w, h, d):
                return cx, cy, cz
                
        return None

    def evaluate_bin_fit(bin_idx, w, h, d):
        """
        Returns (fits, score) where lower score is better (Best Fit).
        Score can be the 'wasted space' or just the max coordinate used.
        Let's use the max extent in x, y, z as a proxy for 'tightness'.
        Actually, Best Fit usually means the position that leaves the least room.
        A simple proxy: minimize the sum of coordinates x+y+z? 
        Or minimize the volume of the remaining space in the bin?
        Let's try minimizing the maximum coordinate (x+y+z) to pack tightly.
        """
        pos = find_position(bin_idx, w, h, d)
        if pos is None:
            return False, float('inf')
        x, y, z = pos
        # Heuristic: prefer positions closer to the origin (0,0,0)
        # This tends to fill the bin from the bottom-left corner up.
        score = x + y + z
        return True, score

    def place_box(w, h, d):
        # 1. Try to place in existing bins (Best Fit)
        best_bin_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i in range(len(bins)):
            fits, score = evaluate_bin_fit(i, w, h, d)
            if fits and score < best_score:
                best_score = score
                best_bin_idx = i
                best_pos = find_position(i, w, h, d) # Recalculate to be safe, or store it
                
        if best_bin_idx != -1:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            bin_volumes[best_bin_idx] += w * h * d
            return best_bin_idx
        
        # 2. If no existing bin works, open a new bin
        new_bin_idx = len(bins)
        bins.append([])
        bin_volumes.append(w * h * d)
        # Place at (0,0,0)
        bins[new_bin_idx].append((0, 0, 0, w, h, d))
        return new_bin_idx

    # Process boxes
    placements = [None] * n
    bin_assignments = [0] * n
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a new bin each, or just break?
            # If we break, we have an invalid solution (missing boxes).
            # We must place all boxes.
            # In case of timeout, we can do a very fast greedy for the rest.
            pass
        
        b = place_box(w, h, d)
        bin_assignments[idx] = b
        # Get the position from the last item in the bin? 
        # Since we just appended, it's the last one.
        pos = bins[b][-1]
        placements[idx] = (b, pos[0], pos[1], pos[2], w, h, d)
        
        # Early exit check for time
        if time.time() > deadline:
             # We need to place the rest. 
             # To ensure validity and finish, we can just dump remaining into new bins.
             # This will be suboptimal but valid.
             remaining = [i for i in range(n) if placements[i] is None]
             # Wait, the loop is over sorted boxes. 
             # If we break here, we haven't placed the rest.
             # Let's implement a fast fallback for the rest.
             # Actually, the loop iterates `indexed_boxes`. 
             # If we run out of time, we should stop searching for optimal positions 
             # and just place remaining boxes in new bins or first available spot quickly.
             # For simplicity and safety, if we hit deadline, we can just place remaining 
             # in new bins. 
             
             # Get indices of unplaced boxes
             # Since we are in the middle of the loop, we can break and handle the rest.
             break
             
    # Handle any unplaced boxes if we broke out early
    if time.time() > deadline:
        # Find unplaced
        unplaced_indices = [i for i in range(n) if placements[i] is None]
        # For each unplaced, assign to a new bin at (0,0,0)
        # This guarantees validity but poor score.
        for i in unplaced_indices:
            w, h, d = boxes[i]
            b = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placements[i] = (b, 0, 0, 0, w, h, d)
            
    return placements
