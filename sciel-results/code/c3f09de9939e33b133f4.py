"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

def pack(boxes, container):
    """
    3D bin packing using a hybrid approach:
    1. Greedy Best-Fit Decreasing with extreme-point placement.
    2. Local search refinement to reduce bin count and improve last-bin fill.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    if n == 0:
        return []
    
    # Safety margin: aim to finish by 110s
    deadline = time.time() + 110.0
    
    # --- Helper Functions ---
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) without overlap and within container."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Check overlap
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def find_positions(bin_boxes, w, h, d):
        """
        Find all valid (x, y, z) positions for a box in a bin.
        Returns list of positions sorted by heuristic (lower is better).
        Uses extreme points (corners defined by existing boxes).
        """
        positions = []
        
        # Generate candidate positions based on existing boxes and container boundaries
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if bx + bw <= W:
                x_candidates.add(bx + bw)
            if by + bh <= H:
                y_candidates.add(by + bh)
            if bz + bd <= D:
                z_candidates.add(bz + bd)
        
        # Filter candidates that are within bounds
        x_candidates = [x for x in x_candidates if 0 <= x and x + w <= W]
        y_candidates = [y for y in y_candidates if 0 <= y and y + h <= H]
        z_candidates = [z for z in z_candidates if 0 <= z and z + d <= D]
        
        # Try all combinations
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer lower z (bottom), then lower y (back), then lower x (left)
                        # This encourages compact packing towards the origin
                        score = z * 10000 + y * 100 + x
                        positions.append((score, x, y, z))
        
        positions.sort()
        return [(x, y, z) for (_, x, y, z) in positions]
    
    def compute_score(placements):
        """
        Compute the score for a set of placements.
        placements: list of (bin_id, x, y, z, w, h, d)
        Score: mean B + 0.1*(1 - last-bin occupied volume fraction)
        Here B is the number of occupied bins (since we evaluate single instance, mean B = B).
        """
        if not placements:
            return 0.0
        
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        
        if B == 0:
            return 0.0
            
        max_bin = max(bin_labels)
        
        # Calculate volume in the last bin
        last_vol = 0
        for p in placements:
            if p[0] == max_bin:
                last_vol += p[4] * p[5] * p[6]
        
        fraction = last_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - fraction)

    # --- Phase 1: Greedy Construction ---
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indices = list(range(n))
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    indices.sort(key=lambda i: -volumes[i])
    
    # Bins: list of lists of (x, y, z, w, h, d)
    bins = []
    # placement_map[i] = (bin_id, x, y, z, w, h, d)
    placement_map = [None] * n
    
    for idx in indices:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins first
        # Heuristic: Try bins in order of decreasing current volume (Best Fit)
        # We need to calculate volume for each bin to sort them.
        if bins:
            bin_vols = []
            for bi in range(len(bins)):
                vol = sum(bw*bh*bd for (_, _, _, bw, bh, bd) in bins[bi])
                bin_vols.append((vol, bi))
            bin_vols.sort(reverse=True) # Highest volume first
            
            for _, bi in bin_vols:
                positions = find_positions(bins[bi], w, h, d)
                if positions:
                    # Pick the first position (best heuristic)
                    x, y, z = positions[0]
                    bins[bi].append((x, y, z, w, h, d))
                    placement_map[idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open a new bin
            new_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[idx] = (new_bin, 0, 0, 0, w, h, d)
        
        # Check deadline
        if time.time() > deadline - 30: # Leave 30s for local search
            break
    
    # Handle any unplaced boxes
    for idx in range(n):
        if placement_map[idx] is None:
            new_bin = len(bins)
            w, h, d = boxes[idx]
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[idx] = (new_bin, 0, 0, 0, w, h, d)
    
    # Convert to internal list for local search: [(bin, x, y, z, w, h, d), ...]
    # We keep track of original index implicitly by position in the list? 
    # No, better to keep a list of dicts or tuples with original index.
    # Let's use a list of 7-tuples (bin, x, y, z, w, h, d) and a separate array for original indices?
    # Actually, the final output needs to be in original order.
    # Let's store placements as a list where index i corresponds to original box i.
    # But for local search, it's easier to have a mutable list of items.
    
    # Let's create a list of [bin, x, y, z, w, h, d] for each original box index.
    current_placements = []
    for i in range(n):
        current_placements.append(list(placement_map[i]))
    
    current_score = compute_score(current_placements)
    best_placements = [p[:] for p in current_placements]
    best_score = current_score
    
    # --- Phase 2: Local Search Refinement ---
    
    # We will try to move boxes to other bins to reduce the number of bins.
    # If we reduce the number of bins, the score drops significantly.
    # If we can't reduce bins, we try to improve the fill of the last bin.
    
    def get_bin_contents(placements):
        """Rebuild bins from a list of placements."""
        b = {}
        for p in placements:
            bidx = p[0]
            if bidx not in b:
                b[bidx] = []
            # p[1..6] are x,y,z,w,h,d
            b[bidx].append((p[1], p[2], p[3], p[4], p[5], p[6]))
        return b

    max_iterations = 20000
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 5000:
            break
            
        # Pick a random box to move
        i = random.randint(0, n - 1)
        bin_id, x, y, z, w, h, d = current_placements[i]
        
        # Get current bins
        bins_dict = get_bin_contents(current_placements)
        
        # Get other bins
        other_bins = [b for b in bins_dict.keys() if b != bin_id]
        
        if not other_bins:
            continue
            
        # Shuffle other bins to try different targets
        random.shuffle(other_bins)
        
        moved = False
        for target_bin in other_bins:
            # Temporarily remove box i from its current bin for the check
            # Actually, we just check if it fits in target_bin.
            # If it fits, we move it.
            
            # Find a position in target_bin
            positions = find_positions(bins_dict[target_bin], w, h, d)
            if positions:
                # Pick a position
                # Heuristic: Pick the position that minimizes the resulting score?
                # For now, pick the first one (most compact).
                nx, ny, nz = positions[0]
                
                # Simulate move
                new_placements = [p[:] for p in current_placements]
                new_placements[i] = [target_bin, nx, ny, nz, w, h, d]
                
                # Check if this move reduces the number of bins
                old_bins = set(p[0] for p in current_placements)
                new_bins = set(p[0] for p in new_placements)
                
                # If the source bin becomes empty, B decreases.
                # If not, B stays same, but we might improve last bin fill.
                
                new_score = compute_score(new_placements)
                
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    best_placements = new_placements
                    current_placements = new_placements
                    moved = True
                    no_improve_count = 0
                    break
                elif new_score < current_score - 1e-9:
                    # Accept if it improves current state (hill climbing)
                    current_placements = new_placements
                    moved = True
                    no_improve_count = 0
                    break
                else:
                    no_improve_count += 1
        
        if not moved:
            no_improve_count += 1

    # Return the best found
    final_placements = best_placements
    
    # Format output
    result = []
    for p in final_placements:
        result.append(tuple(p))
        
    return result

# EVOLVE-BLOCK-END
