import time
import random
import sys
import math

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(j+1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
                    if count >= 4:
                        return False
    return True

def _compute_score(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    Returns the size of the largest such subset.
    """
    n = len(points)
    if n == 0:
        return 0
        
    # Precompute collinearity masks
    # collinear[i][j] is a bitmask of points k such that i, j, k are collinear.
    # Since i,j,k are collinear, k is on the line through i and j.
    # We only care about k > j or k < i to avoid duplicates? No, bitmask handles it.
    # But for the "forbidden" logic, we need to know if adding p creates a triple with existing points.
    
    # Let's precompute for each pair (i, j), the set of other points on the line.
    # line_mask[i][j] includes i and j? No, just the others.
    # Actually, let's just store the set of indices.
    # n is small (20), so lists/sets are fine, but bitmasks are faster for intersection.
    
    # Mask for each pair (i, j) of points that are collinear with them.
    # mask[i][j] = sum(1 << k for k in range(n) if k != i and k != j and collinear(i, j, k))
    
    masks = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            m = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    m |= (1 << k)
            masks[i][j] = m
            masks[j][i] = m

    best = 0
    
    def backtrack(current_mask, candidates_mask):
        nonlocal best
        
        # Pruning 1: Even if we take all candidates, we can't beat best
        # bit_count is fast
        if (current_mask.bit_count() + candidates_mask.bit_count()) <= best:
            return
            
        if candidates_mask == 0:
            if current_mask.bit_count() > best:
                best = current_mask.bit_count()
            return

        # Heuristic: Pick a candidate
        # Pick the first set bit
        # Or pick the one that is in the most "potential conflicts" with other candidates?
        # Let's just pick the lowest index for now, or random?
        # Picking the one with the highest degree in the "conflict hypergraph" might prune better.
        # Degree of p: number of pairs (q, r) in candidates such that p, q, r are collinear?
        # That's expensive to compute on the fly.
        # Let's stick to lowest index for simplicity and determinism in pruning.
        
        # Get first candidate
        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        p_bit = 1 << p_idx
        
        # Branch 1: Exclude p_idx
        backtrack(current_mask, candidates_mask & ~p_bit)
        
        # Branch 2: Include p_idx
        # Check if p_idx can be added to current_mask
        # It cannot be added if it forms a triple with any two points in current_mask.
        # i.e., exists i, j in current_mask such that p_idx, i, j are collinear.
        # This is equivalent to: current_mask & masks[p_idx][k] != 0 for some k in current_mask?
        # No. If p_idx, i, j are collinear, then j is in masks[p_idx][i].
        # So we need to check if there exists i in current_mask such that (current_mask & masks[p_idx][i]) != 0.
        # And i must be distinct from the other point? Yes, masks exclude i and p.
        
        valid_to_add = True
        # Iterate through bits in current_mask
        cm = current_mask
        while cm:
            i_bit = cm & -cm
            i_idx = i_bit.bit_length() - 1
            # Check if there is any j in current_mask such that p, i, j are collinear
            # j must be in masks[p_idx][i_idx] AND j must be in current_mask
            # Note: masks[p_idx][i_idx] does not contain i_idx or p_idx.
            # So if (current_mask & masks[p_idx][i_idx]) != 0, then such a j exists.
            if (current_mask & masks[p_idx][i_idx]) != 0:
                valid_to_add = False
                break
            cm ^= i_bit
            
        if valid_to_add:
            # New current mask
            new_current_mask = current_mask | p_bit
            
            # New candidates mask
            # We remove p_bit from candidates.
            # We also remove any candidate q that forms a triple with p_idx and any point in new_current_mask.
            # If p_idx, q, r are collinear and r is in new_current_mask, then q cannot be added later.
            # So q is forbidden if q is in masks[p_idx][r] for any r in new_current_mask.
            
            forbidden_mask = 0
            ncm = new_current_mask
            while ncm:
                r_bit = ncm & -ncm
                r_idx = r_bit.bit_length() - 1
                if r_idx == p_idx:
                    # masks[p_idx][p_idx] is 0 anyway
                    pass
                forbidden_mask |= masks[p_idx][r_idx]
                ncm ^= r_bit
                
            new_candidates_mask = (candidates_mask & ~p_bit) & ~forbidden_mask
            
            backtrack(new_current_mask, new_candidates_mask)

    backtrack(0, (1 << n) - 1)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
        
    def generate_initial_parabola(offset_x, offset_y, span):
        pts = []
        # Generate points on a parabola y = x^2 + offset
        # Perturb slightly to avoid 4 collinear? Parabola has no 3 collinear?
        # Actually, any line intersects a parabola at at most 2 points.
        # So a set of points on a parabola has NO 3 collinear points.
        # Score will be 20. We want to MINIMIZE score.
        # So we want to ADD collinear triples.
        # Strategy: Start with random points, then optimize.
        # Or start with points that have many collinear triples but no 4.
        # "Grid with holes" or "perturbed grid" is better.
        
        # Let's try a "structured random" approach.
        # Pick 20 random points, ensure no 4 collinear.
        pass
        
    def generate_initial_random():
        pts = []
        attempts = 0
        while len(pts) < 20:
            p = random_point()
            if p in pts:
                continue
            # Check if adding p creates a 4-collinear
            valid = True
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx * (p[1] - y1) == dy * (p[0] - x1):
                        # Count points on this line
                        count = 2
                        for k in range(len(pts)):
                            if k == i or k == j:
                                continue
                            x3, y3 = pts[k]
                            if dx * (y3 - y1) == dy * (x3 - x1):
                                count += 1
                        if count >= 3:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                pts.append(p)
            attempts += 1
            if attempts > 10000:
                # Fallback: just pick random distinct points and hope (unlikely to be invalid if sparse)
                # Or force break
                break
        # If we didn't get 20, fill with random distinct
        while len(pts) < 20:
            p = random_point()
            if p not in pts:
                pts.append(p)
        # Validate
        if not _check_no_four_collinear(pts):
            # Repair? Just regenerate?
            return generate_initial_random()
        return pts

    # Initial configs
    best_points = generate_initial_random()
    best_score = _compute_score(best_points)
    
    current_points = best_points[:]
    current_score = best_score
    
    T = 5.0
    T_min = 0.1
    alpha = 0.995
    tabu_list = []
    tabu_tenure = 50
    
    moves = 0
    
    while time.time() < deadline:
        # Simulated Annealing step
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Generate new position
        # Local move: small step, or global random
        if random.random() < 0.5:
            # Local
            nx = old_p[0] + random.randint(-5, 5)
            ny = old_p[1] + random.randint(-5, 5)
            nx = max(-100, min(100, nx))
            ny = max(-100, min(100, ny))
            new_p = [nx, ny]
        else:
            new_p = random_point()
            
        if new_p == old_p:
            continue
            
        if new_p in current_points:
            continue
            
        # Check Tabu
        # Simple tabu: (idx, new_p) shouldn't have been the previous move for idx
        # Or just keep a list of recent states?
        # Let's use a simple list of (idx, p)
        is_tabu = any(item[0] == idx and item[1] == new_p for item in tabu_list)
        
        if is_tabu and current_score >= best_score:
            continue # Skip tabu move unless it improves best (aspiration)
            
        # Test validity
        current_points[idx] = new_p
        
        if not _check_no_four_collinear(current_points):
            current_points[idx] = old_p
            continue
            
        # Compute score
        new_score = _compute_score(current_points)
        
        # Update Tabu
        tabu_list.append((idx, new_p))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
        # Accept/Reject
        delta = new_score - current_score
        if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = current_points[:]
                # Restart T on improvement?
                # T = 5.0 
        else:
            current_points[idx] = old_p
            # If rejected, we didn't move, so we don't add to tabu? 
            # The tabu should prevent returning to a recently visited state.
            # If we rejected, we are still in old state. The "move" to new_p was considered.
            # So adding to tabu is correct to avoid considering it again immediately.
            
        T *= alpha
        moves += 1
        
        # Check time every 1000 moves
        if moves % 1000 == 0:
            pass # Just a checkpoint
            
    return best_points

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(f"Score: {_compute_score(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
