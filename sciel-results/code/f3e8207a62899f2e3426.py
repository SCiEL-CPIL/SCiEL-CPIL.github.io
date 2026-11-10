# EVOLVE-BLOCK-START
import numpy as np
import time

def _is_valid(points):
    """Check validity constraints."""
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check pairwise squared distances
    for i in range(33):
        for j in range(i + 1, 33):
            d2 = (points[i, 0] - points[j, 0])**2 + (points[i, 1] - points[j, 1])**2
            if d2 < 1e-12:
                return False
    # Check no three collinear
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                val = abs(points[i, 0] * (points[j, 1] - points[k, 1]) +
                          points[j, 0] * (points[k, 1] - points[i, 1]) +
                          points[k, 0] * (points[i, 1] - points[j, 1]))
                if val < 1e-9:
                    return False
    return True

def _count_convex_7gons(points):
    """
    Count the number of 7-point subsets in convex position.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    C = 0
    for i in range(n - 6):
        pivot = pts[i]
        later = pts[i + 1:]
        angles = np.arctan2(later[:, 1] - pivot[1], later[:, 0] - pivot[0])
        sorted_later = later[np.argsort(angles)]
        m = len(sorted_later)
        if m < 6:
            continue
        
        # dp[l][i][j]
        # l is length of chain (points from sorted_later)
        # i, j are indices in sorted_later, i < j
        # We need chains of length 6.
        # dp[2][(i,j)] = 1 if turn(pivot, sl[i], sl[j]) > 1e-9
        # dp[l][(i,j)] = sum dp[l-1][(h,i)] for h < i where turn(sl[h], sl[i], sl[j]) > 1e-9
        
        # Initialize dp for l=2
        dp = {}
        for j in range(1, m):
            for i in range(j):
                cross = (sorted_later[i, 0] - pivot[0]) * (sorted_later[j, 1] - sorted_later[i, 1]) - \
                        (sorted_later[i, 1] - pivot[1]) * (sorted_later[j, 0] - sorted_later[i, 0])
                if cross > 1e-9:
                    dp[(2, i, j)] = 1
                else:
                    dp[(2, i, j)] = 0
        
        # Build up to l=6
        for l in range(3, 7):
            for j in range(2, m):
                for i in range(1, j):
                    total = 0
                    for h in range(i):
                        prev_val = dp.get((l-1, h, i), 0)
                        if prev_val == 0:
                            continue
                        cross = (sorted_later[i, 0] - sorted_later[h, 0]) * (sorted_later[j, 1] - sorted_later[i, 1]) - \
                                (sorted_later[i, 1] - sorted_later[h, 1]) * (sorted_later[j, 0] - sorted_later[i, 0])
                        if cross > 1e-9:
                            total += prev_val
                    dp[(l, i, j)] = total
        
        # Sum valid chains with closing turn
        for j in range(5, m):
            for i in range(4, j):
                val = dp.get((6, i, j), 0)
                if val == 0:
                    continue
                # Closing turn: sl[j] -> pivot -> sl[i]
                cross = (pivot[0] - sorted_later[j, 0]) * (sorted_later[i, 1] - pivot[1]) - \
                        (pivot[1] - sorted_later[j, 1]) * (sorted_later[i, 0] - sorted_later[j, 0]) # Wait, formula check
                # p=sl[j], q=pivot, r=sl[i]
                # (qx-px)(ry-qy) - (qy-py)(rx-qx)
                # qx-px = pivot[0]-sl[j,0]
                # qy-py = pivot[1]-sl[j,1]
                # rx-qx = sl[i,0]-pivot[0]
                # ry-qy = sl[i,1]-pivot[1]
                cross = (pivot[0] - sorted_later[j, 0]) * (sorted_later[i, 1] - pivot[1]) - \
                        (pivot[1] - sorted_later[j, 1]) * (sorted_later[i, 0] - pivot[0])
                if cross > 1e-9:
                    C += val
    return C

def _gen_S(k):
    """
    Generate 2^(k-2) points with no convex k-gon.
    """
    if k == 3:
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    
    A = _gen_S(k-1)
    B = _gen_S(k-1)
    
    # Scale by 0.5
    A_scaled = A * 0.5
    B_scaled = B * 0.5
    
    # Shift A to top, B to bottom
    A_shifted = A_scaled + np.array([0.0, 0.5])
    B_shifted = B_scaled + np.array([0.0, 0.0])
    
    return np.vstack([A_shifted, B_shifted])

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    
    # Generate 32 points with no convex 7-gon
    pts_32 = _gen_S(7)
    
    # Add small perturbations to ensure general position
    np.random.seed(42)
    perturbations = np.random.uniform(-1e-4, 1e-4, (32, 2))
    pts_32 = pts_32 + perturbations
    
    # Clamp to [0,1]
    pts_32 = np.clip(pts_32, 0.0, 1.0)
    
    # Try several candidate positions for the 33rd point
    candidates = [
        np.array([0.5, 0.5]),
        np.array([0.5, 0.25]),
        np.array([0.5, 0.75]),
        np.array([0.25, 0.5]),
        np.array([0.75, 0.5]),
        np.array([0.1, 0.1]),
        np.array([0.9, 0.9]),
        np.array([0.1, 0.9]),
        np.array([0.9, 0.1]),
        np.mean(pts_32, axis=0)
    ]
    
    best_points = None
    best_count = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 60:
            break
            
        # Add small perturbation to candidate as well
        cand_pert = cand + np.random.uniform(-1e-4, 1e-4, 2)
        cand_pert = np.clip(cand_pert, 0.0, 1.0)
        
        points = np.vstack([pts_32, cand_pert.reshape(1, 2)])
        
        if not _is_valid(points):
            continue
            
        count = _count_convex_7gons(points)
        
        if count < best_count:
            best_count = count
            best_points = points.copy()
            
    if best_points is None:
        # Fallback
        best_points = np.vstack([pts_32, np.array([0.5, 0.5])])
        
    return best_points

# EVOLVE-BLOCK-END
