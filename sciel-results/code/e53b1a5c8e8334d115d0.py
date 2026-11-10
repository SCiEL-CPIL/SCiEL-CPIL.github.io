import numpy as np
import time
import random
from itertools import combinations

def compute_all_dets(points):
    """Compute absolute doubled areas for all triples (i<j<k)."""
    n = points.shape[0]
    dets = np.zeros((n*(n-1)*(n-2)//6,), dtype=np.int32)
    idx = 0
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                dets[idx] = abs(det)
                idx += 1
    return dets

def compute_multiplicity(dets):
    """Compute the maximum multiplicity of any positive doubled area."""
    valid = dets[dets > 0]
    if len(valid) == 0:
        return 0
    counts = np.bincount(valid)
    return int(counts.max())

def fast_score(points):
    """Fast scoring function using vectorized computation."""
    n = points.shape[0]
    if n < 3:
        return 0
    # Use the O(n^3) approach but vectorized for speed
    x = points[:, 0].astype(np.int64)
    y = points[:, 1].astype(np.int64)
    # For each pair (i,j), compute determinant with all k
    # det(i,j,k) = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
    # = (x_j - x_i)*y_k - (y_j - y_i)*x_k - (x_j - x_i)*y_i + (y_j - y_i)*x_i
    # This is linear in x_k, y_k
    
    # Build all triplets vectorized
    n = points.shape[0]
    # Create index arrays for i < j < k
    ii, jj, kk = np.indices((n, n, n))
    mask = (ii < jj) & (jj < kk)
    ii = ii[mask]
    jj = jj[mask]
    kk = kk[mask]
    
    det = (points[jj, 0].astype(np.int64) - points[ii, 0].astype(np.int64)) * \
          (points[kk, 1].astype(np.int64) - points[ii, 1].astype(np.int64)) - \
          (points[jj, 1].astype(np.int64) - points[ii, 1].astype(np.int64)) * \
          (points[kk, 0].astype(np.int64) - points[ii, 0].astype(np.int64))
    
    valid = np.abs(det)
    valid = valid[valid > 0]
    if len(valid) == 0:
        return 0
    counts = np.bincount(valid)
    return int(counts.max())

def delta_score(points, old_dets, changed_idx, new_point):
    """
    Compute new score after changing points[changed_idx] to new_point.
    Only triples involving changed_idx change.
    Returns (new_multiplicity, new_dets_array)
    """
    n = points.shape[0]
    new_dets = old_dets.copy()
    
    # Find indices of triples involving changed_idx
    # Triple (i,j,k) with i<j<k contains changed_idx
    # We need to recompute det for triples where i==changed_idx or j==changed_idx or k==changed_idx
    
    # First, let's figure out the linear index for each triple
    # Triple (i,j,k) has index = sum of (i-1)*(n-1) + ... actually let's just recompute
    # All triples involving changed_idx
    new_point = np.array(new_point, dtype=np.int64)
    old_point = points[changed_idx].astype(np.int64)
    
    # Build a map from (i,j,k) to linear index
    # Actually, let's just recompute all dets for triples involving changed_idx
    # and update new_dets accordingly
    
    # Create a dictionary mapping (i,j,k) to index in dets array
    idx_map = {}
    idx = 0
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                idx_map[(i,j,k)] = idx
                idx += 1
    
    # Recompute for triples involving changed_idx
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                if i == changed_idx or j == changed_idx or k == changed_idx:
                    # Get the three points
                    p1_idx = min(i, j, k)
                    p2_idx = max(min(i, j), min(i, k), min(j, k))
                    p3_idx = max(i, j, k)
                    
                    # Actually, we just need the three points in order
                    pts = [points[m] for m in [i, j, k]]
                    pts[changed_idx if changed_idx in [i,j,k] else -1] = new_point
                    # Wait, this is wrong. Let me redo.
                    
                    # Get the actual point indices
                    idxs = [i, j, k]
                    # Replace changed_idx with new_point
                    x_vals = []
                    y_vals = []
                    for m in idxs:
                        if m == changed_idx:
                            x_vals.append(new_point[0])
                            y_vals.append(new_point[1])
                        else:
                            x_vals.append(points[m, 0])
                            y_vals.append(points[m, 1])
                    
                    det = (x_vals[1] - x_vals[0]) * (y_vals[2] - y_vals[0]) - \
                          (y_vals[1] - y_vals[0]) * (x_vals[2] - x_vals[0])
                    new_dets[idx_map[(i,j,k)]] = abs(det)
    
    new_mult = compute_multiplicity(new_dets)
    return new_mult, new_dets

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    random.seed(42)
    np.random.seed(42)
    
    candidates = []
    
    # Rectangular Grids W x H such that W*H = 30
    for W in [1, 2, 3, 5, 6, 10, 15, 30]:
        H = 30 // W
        if W * H != 30:
            continue
        pts = []
        for i in range(W):
            for j in range(H):
                x = i - W // 2
                y = j - H // 2
                pts.append((x, y))
        
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(np.array(pts, dtype=np.int64))
    
    # Add a "clustered" start: points close to each other
    for _ in range(8):
        pts = []
        base_x = random.randint(-20, 20)
        base_y = random.randint(-20, 20)
        spread = random.randint(3, 12)
        while len(pts) < N:
            p = (base_x + random.randint(-spread, spread), 
                 base_y + random.randint(-spread, spread))
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in pts:
                pts.append(p)
        if len(pts) == N:
            candidates.append(np.array(pts, dtype=np.int64))
    
    # Add random starts
    for _ in range(8):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=np.int64))
    
    # Add structured starts: points on a circle-like pattern
    for _ in range(5):
        pts = []
        cx = random.randint(-5, 5)
        cy = random.randint(-5, 5)
        r = random.randint(8, 25)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        for a in angles:
            x = int(round(cx + r * np.cos(a)))
            y = int(round(cy + r * np.sin(a)))
            # Add jitter to avoid collinearities
            x += random.randint(-1, 1)
            y += random.randint(-1, 1)
            x = max(-30, min(30, x))
            y = max(-30, min(30, y))
            p = (x, y)
            if p not in pts:
                pts.append(p)
        # Fill any duplicates
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=np.int64))
    
    # Add structured starts: points on a lattice with perturbations
    for _ in range(5):
        pts = []
        grid_size = random.randint(5, 8)
        step = random.randint(3, 6)
        ox = random.randint(-10, 10)
        oy = random.randint(-10, 10)
        for i in range(grid_size):
            for j in range(grid_size):
                x = ox + i * step + random.randint(-1, 1)
                y = oy + j * step + random.randint(-1, 1)
                x = max(-30, min(30, x))
                y = max(-30, min(30, y))
                p = (x, y)
                if p not in pts and len(pts) < N:
                    pts.append(p)
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=np.int64))
    
    # Evaluate candidates and pick the best few for local search
    scored_candidates = []
    for cand in candidates:
        s = fast_score(cand)
        scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score = -1
    best_global_pts = None
    best_global_dets = None
    
    # Run local search on the top 4 initial candidates
    for initial_score, initial_pts in scored_candidates[:4]:
        if time.time() > deadline:
            break
            
        current_pts = initial_pts.copy()
        current_score = initial_score
        current_dets = compute_all_dets(current_pts)
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_pts = current_pts.copy()
            best_global_dets = current_dets.copy()
        
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.005
        alpha = 0.995
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 2.0:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                best_global_dets = current_dets.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            idx = np.random.randint(N)
            
            if random.random() < 0.75:
                # Small move
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_p = tuple(current_pts[idx] + np.array([dx, dy], dtype=np.int64))
            else:
                # Larger move
                new_p = (int(random.randint(-30, 30)), int(random.randint(-30, 30)))
                
            if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                continue
                
            is_distinct = True
            for i in range(N):
                if i != idx and current_pts[i, 0] == new_p[0] and current_pts[i, 1] == new_p[1]:
                    is_distinct = False
                    break
                    
            if not is_distinct:
                continue
            
            new_score, new_dets = delta_score(current_pts, current_dets, idx, new_p)
            
            if new_score > current_score:
                current_pts[idx] = np.array(new_p, dtype=np.int64)
                current_score = new_score
                current_dets = new_dets
            else:
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / max(T, 1e-10))
                    if random.random() < prob:
                        current_pts[idx] = np.array(new_p, dtype=np.int64)
                        current_score = new_score
                        current_dets = new_dets
                elif random.random() < 0.3:
                    current_pts[idx] = np.array(new_p, dtype=np.int64)
                    current_score = new_score
                    current_dets = new_dets
            
            T = max(T_min, T * alpha)
    
    # Final Polish Phase - very fine-grained moves
    if best_global_pts is not None and time.time() < deadline - 5.0:
        current_pts = best_global_pts.copy()
        current_score = best_global_score
        current_dets = compute_all_dets(current_pts)
        
        T = 0.05
        T_min = 0.001
        alpha = 0.998
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 1.0:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                best_global_dets = current_dets.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            idx = np.random.randint(N)
            
            # Very small moves for polish
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            if dx == 0 and dy == 0:
                continue
            new_p = tuple(current_pts[idx] + np.array([dx, dy], dtype=np.int64))
                
            if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                continue
                
            is_distinct = True
            for i in range(N):
                if i != idx and current_pts[i, 0] == new_p[0] and current_pts[i, 1] == new_p[1]:
                    is_distinct = False
                    break
                    
            if not is_distinct:
                continue
            
            new_score, new_dets = delta_score(current_pts, current_dets, idx, new_p)
            
            if new_score > current_score:
                current_pts[idx] = np.array(new_p, dtype=np.int64)
                current_score = new_score
                current_dets = new_dets
            else:
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / max(T, 1e-10))
                    if random.random() < prob:
                        current_pts[idx] = np.array(new_p, dtype=np.int64)
                        current_score = new_score
                        current_dets = new_dets
                elif random.random() < 0.1:
                    current_pts[idx] = np.array(new_p, dtype=np.int64)
                    current_score = new_score
                    current_dets = new_dets
            
            T = max(T_min, T * alpha)
    
    return [(int(p[0]), int(p[1])) for p in best_global_pts]
