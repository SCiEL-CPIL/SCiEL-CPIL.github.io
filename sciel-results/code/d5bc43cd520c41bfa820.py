"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Extreme Points heuristic with Dominance Pruning.
    Improves upon the basic EP heuristic by:
    1. Pruning dominated extreme points to keep the candidate set small and focused.
    2. Using a Best-Fit-Decreasing strategy with a composite score for bin selection.
    3. Optimizing the overlap check and EP update logic.
    """
    W, H, D = container
    container_volume = W * H * D
    
    # Sort boxes by volume descending, then by max dimension descending
    sorted_indices = sorted(range(len(boxes)), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], max(boxes[i])), reverse=True)
    
    bins = []
    
    def is_valid_pos(bin_boxes, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def prune_dominated(eps):
        """
        Remove points that are dominated by other points in the set.
        P1 dominates P2 if P1.x <= P2.x, P1.y <= P2.y, P1.z <= P2.z and P1 != P2.
        Since we minimize (z, y, x), a lower point is always at least as good.
        """
        eps_list = list(eps)
        if len(eps_list) <= 1:
            return set(eps_list)
        
        # Sort to make dominance check easier? Not strictly necessary for small N.
        # O(N^2) check.
        nondominated = set()
        for i, p1 in enumerate(eps_list):
            dominated = False
            for j, p2 in enumerate(eps_list):
                if i == j:
                    continue
                # Check if p2 dominates p1
                if p2[0] <= p1[0] and p2[1] <= p1[1] and p2[2] <= p1[2] and (p2[0] < p1[0] or p2[1] < p1[1] or p2[2] < p1[2]):
                    dominated = True
                    break
            if not dominated:
                nondominated.add(p1)
        return nondominated

    def generate_new_extreme_points(bin_boxes, x, y, z, w, h, d, current_eps):
        new_candidates = set()
        new_candidates.add((x + w, y, z))
        new_candidates.add((x, y + h, z))
        new_candidates.add((x, y, z + d))
        
        all_points = set(current_eps)
        all_points.update(new_candidates)
        
        final_eps = set()
        for p in all_points:
            px, py, pz = p
            inside = False
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if (bx <= px < bx + bw) and (by <= py < by + bh) and (bz <= pz < bz + bd):
                    inside = True
                    break
            if not inside:
                if 0 <= px <= W and 0 <= py <= H and 0 <= pz <= D:
                    final_eps.add(p)
                    
        # Prune dominated points
        final_eps = prune_dominated(final_eps)
        
        return final_eps

    placements = [None] * len(boxes)
    bin_count = 0
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        
        best_bin_index = -1
        best_pos = None
        best_score = None
        best_bin_volume = 0
        
        for b_idx in range(bin_count):
            bin_boxes = bins[b_idx][0]
            eps = bins[b_idx][1]
            
            bin_best_pos = None
            bin_best_score = None
            
            for (x, y, z) in eps:
                if is_valid_pos(bin_boxes, x, y, z, w, h, d):
                    # Score: Minimize Z, then Y, then X
                    score = (z, y, x)
                    
                    if bin_best_score is None or score < bin_best_score:
                        bin_best_score = score
                        bin_best_pos = (x, y, z)
                        
            if bin_best_pos is not None:
                # Compare across bins
                # Primary: Lowest Z (compactness)
                # Secondary: Prefer bins with more existing volume (fill full bins)
                current_bin_volume = sum(b[3]*b[4]*b[5] for b in bin_boxes)
                
                if best_score is None:
                    best_score = bin_best_score
                    best_bin_index = b_idx
                    best_pos = bin_best_pos
                    best_bin_volume = current_bin_volume
                else:
                    # If scores are equal, pick the bin with more volume (more full)
                    if bin_best_score < best_score:
                        best_score = bin_best_score
                        best_bin_index = b_idx
                        best_pos = bin_best_pos
                        best_bin_volume = current_bin_volume
                    elif bin_best_score == best_score:
                        if current_bin_volume > best_bin_volume:
                            best_bin_index = b_idx
                            best_pos = bin_best_pos
                            best_bin_volume = current_bin_volume
        
        if best_bin_index != -1:
            x, y, z = best_pos
            old_boxes, old_eps = bins[best_bin_index]
            new_box = (x, y, z, w, h, d)
            new_boxes = old_boxes + [new_box]
            new_eps = generate_new_extreme_points(new_boxes, x, y, z, w, h, d, old_eps)
            bins[best_bin_index] = (new_boxes, new_eps)
            placements[idx] = (best_bin_index, x, y, z, w, h, d)
        else:
            new_bin_index = bin_count
            bin_count += 1
            new_box = (0, 0, 0, w, h, d)
            new_boxes = [new_box]
            new_eps = generate_new_extreme_points(new_boxes, 0, 0, 0, w, h, d, {(0,0,0)})
            bins.append((new_boxes, new_eps))
            placements[idx] = (new_bin_index, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
