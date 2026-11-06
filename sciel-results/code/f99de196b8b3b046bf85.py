"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: 145s to be safe within 150s limit
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Preprocess: Sort boxes by volume descending for better B&B pruning
    # We keep original indices to map back to the final placement list
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # 2. Data Structures for B&B
    # Bins are represented as a list of boxes: (x, y, z, w, h, d)
    # We use a recursive B&B approach.
    
    # To speed up spot finding, we maintain a list of "free points" (front corners) for each bin.
    # A free point is a position (x, y, z) where a new box can potentially start.
    # Initial free point for an empty bin is (0,0,0).
    
    def generate_candidates(bin_boxes, w, h, d):
        """
        Generate candidate positions for a new box in a bin.
        We use the 'front corner' heuristic:
        Start with (0,0,0).
        For each existing box, add points:
        (x, y, z+d), (x+w, y, z), (x, y+h, z)
        And combinations of these to fill gaps.
        We filter out invalid positions and duplicates.
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Collect all potential x, y, z coordinates from existing boxes
        # This is a more robust way to find "front" points than just single offsets
        xs = [0, W]
        ys = [0, H]
        zs = [0, D]
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            xs.append(bx)
            xs.append(bx + bw)
            ys.append(by)
            ys.append(by + bh)
            zs.append(bz)
            zs.append(bz + bd)
            
        # Unique and sort
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        zs = sorted(set(zs))
        
        # Generate all combinations of these coordinates as potential start points
        # This is O(K^3) where K is number of unique coords.
        # For 40 boxes, K is small (maybe 10-20), so K^3 is ~1000-8000 checks per bin.
        # This is acceptable for 40 boxes total.
        
        valid_candidates = []
        for cx in xs:
            if cx + w > W: continue
            for cy in ys:
                if cy + h > H: continue
                for cz in zs:
                    if cz + d > D: continue
                    
                    # Check overlap with existing boxes
                    overlap = False
                    for bx, by, bz, bw, bh, bd in bin_boxes:
                        if cx < bx + bw and bx < cx + w and \
                           cy < by + bh and by < cy + h and \
                           cz < bz + bd and bz < cz + d:
                            overlap = True
                            break
                    if not overlap:
                        # Sort key: prefer lower z, then y, then x for compactness
                        valid_candidates.append((cz, cy, cx, cx, cy, cz)) # (sort_key, x, y, z)
        
        # Sort by compactness (lowest z, then y, then x)
        if valid_candidates:
            valid_candidates.sort()
            # Return the best few? Or all?
            # For B&B, we usually want to try the "best" placement first to find a solution quickly.
            # Returning all valid candidates allows the B&B to explore alternatives if the first one fails later.
            # However, exploring all valid candidates for every box in every branch is expensive.
            # Strategy: Only return the single best (most compact) candidate?
            # No, that's greedy. B&B needs to explore.
            # But 40 boxes is small. Let's return the top 5 most compact candidates to limit branching factor.
            return [ (x, y, z) for _, x, y, z in valid_candidates[:5] ]
        return []

    # Global best solution
    best_bin_count = float('inf')
    best_placement = None
    best_last_bin_volume = 0.0
    
    # Current state
    # bins: list of lists of (x, y, z, w, h, d)
    # current_bin_idx: index of the bin we are currently filling (if we are filling sequentially)
    # Actually, standard B&B for bin packing:
    # State: (box_index, bins_state)
    # bins_state is a tuple of tuples to be hashable? No, we modify in place and backtrack.
    
    # We will use a recursive function.
    # To optimize, we'll try to fill bin 0, then bin 1, etc.
    # If we open a new bin, we start filling it.
    
    # Optimizations:
    # 1. Symmetry breaking: If we open a new bin, we don't need to try placing the box in an empty bin if there's already an empty bin.
    # 2. Lower bound: If current_bin_count >= best_bin_count, prune.
    
    def solve(idx, bins):
        nonlocal best_bin_count, best_placement, best_last_bin_volume, deadline
        
        # Time check
        if time.time() > deadline:
            return False # Signal to stop
        
        # Pruning: If we already have more bins than the best known, stop
        if len(bins) > best_bin_count:
            return False
        
        # Base case: All boxes placed
        if idx == n:
            # Update best
            # Calculate score components
            b_count = len(bins)
            # Find last bin occupied volume
            # The "last bin" is the one with the largest label. Since we create bins sequentially, it's bins[-1] if it's not empty.
            # But what if the last bin is empty? The problem says "largest label defines the last bin".
            # If we have bins [0, 1, 2] and bin 2 is empty, B=3? No, B is number of OCCUPIED bins.
            # "B is the number of occupied bins".
            # "last-bin occupied volume fraction" -> volume of the bin with the largest label (among occupied ones? or just the last index used?).
            # Contract: "largest label defines the last bin".
            # If I use labels 0, 1, 3, the last bin is 3.
            # In my implementation, labels are 0..len(bins)-1.
            # If the last bin (index len-1) is empty, it shouldn't be counted in B?
            # "B is the number of occupied bins".
            # So if bins[-1] is empty, B = len(bins)-1.
            # And "last-bin" would be bins[-2]?
            # Let's assume we only keep non-empty bins in the list `bins`.
            # So bins[-1] is always the last occupied bin.
            
            if b_count < best_bin_count:
                best_bin_count = b_count
                # Deep copy placements
                # We need to reconstruct placements from bins
                # But we need to map back to original indices.
                # We stored (x, y, z, w, h, d) in bins. We lost the original index.
                # We need to track which original box went into which position.
                
                # Let's modify the state to track this.
                # Instead of just storing geometry in bins, we store (orig_idx, x, y, z, w, h, d).
                
                # Reconstruct
                placements = [None] * n
                for b_idx, bin_boxes in enumerate(bins):
                    for item in bin_boxes:
                        orig_idx = item[0]
                        x, y, z, w, h, d = item[1], item[2], item[3], item[4], item[5], item[6]
                        placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                
                best_placement = placements
                # Calculate last bin volume
                last_bin_vol = sum(item[4]*item[5]*item[6] for item in bins[-1])
                best_last_bin_volume = last_bin_vol
            return True # Found a valid solution, but keep searching for better? 
            # Actually, if we found a solution with B bins, any solution with B bins is not better in B.
            # The score is B + 0.1*(1 - frac).
            # So if B is the same, we want to maximize last_bin_vol.
            # We should continue searching to see if we can find another solution with same B but better last_bin_vol.
            # But B&B usually stops when it finds a solution with B=1 (perfect).
            # Here, we want to minimize B. If we find B=2, we can still find B=2 with better volume.
            # But finding B=1 is the main goal.
            # Let's just return True to indicate a valid solution was found, and let the caller decide.
            # Actually, for B&B, we usually want to explore all possibilities with B < current best.
            # If we found B=3, and we are exploring a branch that might lead to B=3, we can prune if we already have a B=3 with better volume?
            # This is getting complex. Let's simplify:
            # We will keep searching until we prove B=1 is impossible, or time runs out.
            # If we find a solution with B, we update best.
            # We prune if len(bins) >= best_bin_count.
            # This means if best is 3, we don't explore branches that open a 4th bin.
            # We DO explore branches that might result in 3 bins, to find a better last_bin_volume.
            # So we return False (don't stop) to allow backtracking?
            # No, if we return False, the parent might try other options.
            # If we return True, the parent might stop?
            # Let's just return False always from base case, so the search continues to find better volume or lower B.
            # The pruning `len(bins) > best_bin_count` handles the B part.
            # The volume part is only updated when we hit a base case.
            return False

        # Get current box
        orig_idx, w, h, d, vol = indexed_boxes[idx]
        
        # Try placing in existing bins
        for b_idx in range(len(bins)):
            bin_boxes = bins[b_idx]
            candidates = generate_candidates(bin_boxes, w, h, d)
            
            for (cx, cy, cz) in candidates:
                # Place box
                bins[b_idx].append((orig_idx, cx, cy, cz, w, h, d))
                
                # Recurse
                if solve(idx + 1, bins):
                    # If we found a solution, we could return early if we want the first one?
                    # No, we want the best.
                    pass
                
                # Backtrack
                bins[b_idx].pop()
                
                # Time check
                if time.time() > deadline:
                    return False

        # Try placing in a NEW bin
        # Symmetry breaking: If there are already empty bins, don't create another?
        # In my list `bins`, all bins are non-empty because I only append when placing.
        # So I just need to create a new bin.
        
        # Optimization: If we have already opened `best_bin_count` bins, we can't open more.
        # This is handled by the check at the start of solve().
        
        new_bin = [(orig_idx, 0, 0, 0, w, h, d)]
        bins.append(new_bin)
        
        if solve(idx + 1, bins):
            pass
        
        bins.pop()
        
        return False

    # Initial state
    bins = []
    solve(0, bins)
    
    # If no solution found (should not happen with B&B unless timeout), fallback to greedy
    if best_placement is None:
        # Fallback: Simple greedy FFD
        bins_fallback = []
        placements = [None] * n
        for i, w, h, d, vol in indexed_boxes:
            placed = False
            for b_idx, bin_boxes in enumerate(bins_fallback):
                candidates = generate_candidates(bin_boxes, w, h, d)
                if candidates:
                    cx, cy, cz = candidates[0]
                    bin_boxes.append((i, cx, cy, cz, w, h, d))
                    placements[i] = (b_idx, cx, cy, cz, w, h, d)
                    placed = True
                    break
            if not placed:
                b_idx = len(bins_fallback)
                bins_fallback.append([(i, 0, 0, 0, w, h, d)])
                placements[i] = (b_idx, 0, 0, 0, w, h, d)
        best_placement = placements

    return best_placement
# EVOLVE-BLOCK-END
