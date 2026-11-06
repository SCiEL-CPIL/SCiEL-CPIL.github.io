"""No-isosceles set construction with vectorized validation and local search."""
import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290 second deadline to be safe
    
    best_points = []
    best_count = 0
    
    # Precompute all grid points
    all_grid = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
    
    def compute_d_sq(P):
        """Compute pairwise squared distances for points P (shape n x 2)."""
        n = len(P)
        if n == 0:
            return np.zeros((0, 0), dtype=np.int64)
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        return D_sq
    
    def is_valid_addition(points, p):
        if not points:
            return True
        
        px, py = p
        P = np.array(points, dtype=np.int64)
        n = len(P)
        
        # Squared distances from p to all points in P
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
        
        # Check 1: d(p, a) == d(p, b) for any a != b
        if len(np.unique(d_p)) != n:
            return False
        
        # Squared distances between all pairs in P
        D_sq = compute_d_sq(P)
        
        # Check 2 and 3: For all pairs (i, j) with i != j:
        # d_p[i] != D_sq[i, j] and d_p[j] != D_sq[i, j]
        
        off_diag = ~np.eye(n, dtype=bool)
        
        cond1 = (D_sq == d_p[:, np.newaxis]) & off_diag
        cond2 = (D_sq == d_p[np.newaxis, :]) & off_diag
        
        if np.any(cond1) or np.any(cond2):
            return False
        
        return True

    def is_valid_addition_batch(points, candidates):
        """
        points: list of (x, y)
        candidates: numpy array of shape (M, 2)
        returns: numpy boolean array of shape (M,) indicating which candidates are valid to add
        """
        if not points:
            return np.ones(len(candidates), dtype=bool)
            
        P = np.array(points, dtype=np.int64)
        n = len(P)
        C = candidates
        M = len(C)
        
        # d_c[i, j] = squared distance between candidate i and point j
        diff = C[:, np.newaxis, :] - P[np.newaxis, :, :]
        d_c = np.sum(diff**2, axis=2)  # shape (M, n)
        
        # Check 1: For each candidate i, d_c[i, :] must have unique values
        sorted_d_c = np.sort(d_c, axis=1)
        diff_sorted = np.diff(sorted_d_c, axis=1)
        has_dup_dist = np.any(diff_sorted == 0, axis=1)
        
        # Check 2 and 3:
        D_sq = compute_d_sq(P)
        
        invalid_check2 = np.zeros(M, dtype=bool)
        invalid_check3 = np.zeros(M, dtype=bool)
        
        off_diag = ~np.eye(n, dtype=bool)
        
        for a in range(n):
            # Check 2: d_c[:, a] should not equal any D_sq[a, b] for b != a
            forbidden_a = D_sq[a, off_diag[a]]
            if len(forbidden_a) > 0:
                matches = np.isin(d_c[:, a], forbidden_a)
                invalid_check2 |= matches
            
            # Check 3: d_c[:, a] should not equal any D_sq[b, a] for b != a
            forbidden_a_col = D_sq[off_diag[:, a], a]
            if len(forbidden_a_col) > 0:
                matches = np.isin(d_c[:, a], forbidden_a_col)
                invalid_check3 |= matches
        
        has_invalid_pair = invalid_check2 | invalid_check3
        
        valid_mask = ~has_dup_dist & ~has_invalid_pair
        
        return valid_mask

    def greedy_construct(ordering):
        pts = []
        candidates = list(all_grid)
        if ordering == 'random':
            random.shuffle(candidates)
        elif ordering == 'hilbert':
            # Simple space filling curve approximation: sort by x+y, then x
            # Or actual Hilbert index calculation is complex, let's use a simpler structured order
            # Sort by (x+y, x)
            candidates.sort(key=lambda p: (p[0] + p[1], p[0]))
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(pts, tuple(p)):
                pts.append(tuple(p))
        return pts

    def local_search(initial_pts):
        pts = list(initial_pts)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding any valid point
            pts_set = set(pts)
            mask_not_in = np.array([tuple(c) not in pts_set for c in all_grid], dtype=bool)
            candidates = all_grid[mask_not_in]
            
            if len(candidates) > 0:
                valid_mask = is_valid_addition_batch(pts, candidates)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    idx = valid_indices[0]
                    p = tuple(candidates[idx])
                    pts.append(p)
                    improved = True
            
            # Phase 2: If no point could be added, try swapping
            if not improved:
                # Try 1-for-2 exchange: Remove one point, add two
                n_pts = len(pts)
                swapped = False
                for i in range(n_pts):
                    if time.time() > deadline - 5:
                        return pts
                    
                    old_pt = pts[i]
                    pts_temp = pts[:i] + pts[i+1:]
                    
                    pts_temp_set = set(pts_temp)
                    mask_not_in = np.array([tuple(c) not in pts_temp_set for c in all_grid], dtype=bool)
                    candidates = all_grid[mask_not_in]
                    
                    if len(candidates) == 0:
                        continue
                        
                    valid_mask = is_valid_addition_batch(pts_temp, candidates)
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) >= 2:
                        # Try to find two candidates that are valid with each other
                        # Pick the first two valid candidates c1, c2
                        # Check if c1 and c2 form an isosceles with any point in pts_temp or each other
                        # Actually, is_valid_addition_batch ensures c1 is valid with pts_temp.
                        # We need to check if c1 and c2 are valid with EACH OTHER.
                        # i.e., is {c1, c2} a valid set? Yes, if d(c1, c2) is unique relative to others?
                        # No, we just need to ensure that adding c2 to {c1} U pts_temp is valid.
                        
                        c1_idx = valid_indices[0]
                        c1 = tuple(candidates[c1_idx])
                        
                        # Check if we can add c2 to pts_temp + [c1]
                        # We can reuse is_valid_addition for the single point c2 against pts_temp + [c1]
                        # But we need to find a c2 from the valid_indices list (excluding c1)
                        # that is also valid with c1.
                        
                        # Let's just iterate through a few candidates to find a pair
                        found_pair = False
                        for j in range(1, min(len(valid_indices), 20)): # Check first 20 candidates
                            c2_idx = valid_indices[j]
                            c2 = tuple(candidates[c2_idx])
                            
                            # Check if c2 is valid with pts_temp + [c1]
                            # We can construct the new set and check
                            new_set = pts_temp + [c1]
                            if is_valid_addition(new_set, c2):
                                pts = new_set + [c2]
                                improved = True
                                swapped = True
                                found_pair = True
                                break
                        
                        if found_pair:
                            break
                            
                # If 1-for-2 failed, try 1-for-1 (original logic)
                if not swapped:
                    for i in range(len(pts)):
                        if time.time() > deadline - 5:
                            return pts
                        old_pt = pts[i]
                        pts_temp = pts[:i] + pts[i+1:]
                        
                        pts_temp_set = set(pts_temp)
                        mask_not_in = np.array([tuple(c) not in pts_temp_set for c in all_grid], dtype=bool)
                        candidates = all_grid[mask_not_in]
                        
                        if len(candidates) == 0:
                            continue
                            
                        valid_mask = is_valid_addition_batch(pts_temp, candidates)
                        valid_indices = np.where(valid_mask)[0]
                        
                        if len(valid_indices) > 0:
                            idx = valid_indices[0]
                            p = tuple(candidates[idx])
                            pts = pts_temp + [p]
                            improved = True
                            break
                            
        return pts

    # Run multiple greedy constructions
    max_trials = 50
    
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break
        
        # Alternate between random and structured
        ordering = 'random' if trial % 2 == 0 else 'hilbert'
        pts = greedy_construct(ordering)
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
        # Apply local search if we have time and potential
        if len(pts) >= best_count - 1 and time.time() < deadline - 30:
            improved_pts = local_search(pts)
            if len(improved_pts) > best_count:
                best_points = improved_pts
                best_count = len(improved_pts)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
