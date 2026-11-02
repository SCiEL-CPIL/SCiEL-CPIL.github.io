import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 145.0  # Safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)
_NUM_TRIPLES = len(_triple_i)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _local_search(pts, max_iter=500, time_budget=5.0, t_start=None):
    """
    Direct local search: repeatedly identify the worst (smallest) triangle(s)
    and move the involved points to increase the minimum area.
    This is a structural change from generic SA to targeted optimization.
    """
    if t_start is None:
        t_start = time.time()
    
    pts = pts.copy()
    current_score = _score_points(pts)
    if current_score <= 0:
        return pts, current_score
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        # Time check
        if (time.time() - t_start) > time_budget:
            break
        
        # Compute all areas and find the worst ones
        areas = _compute_areas(pts)
        min_area = np.min(areas)
        
        # Find all triangles with area close to the minimum (within 5%)
        threshold = min_area * 1.05
        worst_idx = np.where(areas <= threshold)[0]
        
        if len(worst_idx) == 0:
            break
        
        # Collect the points involved in the worst triangles
        involved_points = set()
        for idx in worst_idx:
            involved_points.add(_triple_i[idx])
            involved_points.add(_triple_j[idx])
            involved_points.add(_triple_k[idx])
        
        # For each involved point, try moving it in a direction that
        # increases the areas of the worst triangles
        improved = False
        for pt_idx in list(involved_points):
            if (time.time() - t_start) > time_budget:
                break
            
            original_pt = pts[pt_idx].copy()
            
            # Try multiple step sizes and directions
            best_new_pt = None
            best_new_score = current_score
            
            # Compute gradient-like direction: move away from the other two
            # points in each worst triangle containing this point
            dx_total = 0.0
            dy_total = 0.0
            count = 0
            
            for idx in worst_idx:
                ti, tj, tk = _triple_i[idx], _triple_j[idx], _triple_k[idx]
                if ti == pt_idx:
                    # Move away from the midpoint of the other two
                    other1 = pts[tj]
                    other2 = pts[tk]
                    mid = (other1 + other2) / 2
                    dx_total += original_pt[0] - mid[0]
                    dy_total += original_pt[1] - mid[1]
                    count += 1
                elif tj == pt_idx:
                    other1 = pts[ti]
                    other2 = pts[tk]
                    mid = (other1 + other2) / 2
                    dx_total += original_pt[0] - mid[0]
                    dy_total += original_pt[1] - mid[1]
                    count += 1
                elif tk == pt_idx:
                    other1 = pts[ti]
                    other2 = pts[tj]
                    mid = (other1 + other2) / 2
                    dx_total += original_pt[0] - mid[0]
                    dy_total += original_pt[1] - mid[1]
                    count += 1
            
            if count > 0:
                dx_avg = dx_total / count
                dy_avg = dy_total / count
                norm = np.sqrt(dx_avg**2 + dy_avg**2)
                if norm > 1e-10:
                    dx_avg /= norm
                    dy_avg /= norm
            
            # Try several step sizes in the average direction + random perturbations
            for step in [0.01, 0.02, 0.05, 0.1, 0.2]:
                # Direction 1: average gradient
                new_pt = original_pt.copy()
                if count > 0:
                    new_pt[0] += step * dx_avg
                    new_pt[1] += step * dy_avg
                
                # Direction 2: random perturbation
                rand_pt = original_pt + rng.normal(0, step * 0.5, size=2)
                
                for candidate in [new_pt, rand_pt]:
                    candidate = np.clip(candidate, -1.0, 1.0)
                    pts[pt_idx] = candidate
                    new_score = _score_points(pts)
                    if new_score > best_new_score + 1e-10:
                        best_new_score = new_score
                        best_new_pt = candidate.copy()
            
            if best_new_pt is not None:
                pts[pt_idx] = best_new_pt
                current_score = best_new_score
                improved = True
        
        if not improved:
            # Try a larger random kick to escape local minimum
            kick_idx = rng.randint(0, N)
            pts[kick_idx] = np.clip(pts[kick_idx] + rng.normal(0, 0.1, size=2), -1.0, 1.0)
            new_score = _score_points(pts)
            if new_score > current_score:
                current_score = new_score
            else:
                pts[kick_idx] = pts[kick_idx]  # Already applied, check if better
                if current_score <= 0:
                    break
    
    final_score = _score_points(pts)
    return pts, final_score

def _make_initial_guesses():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1)
    
    # 2. 1 center + 12 outer
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    inits.append(p2)
    
    # 3. 1 center + 6 mid + 6 outer
    p3 = np.zeros((N, 2))
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p3[7:, 0] = np.cos(angles_o)
    p3[7:, 1] = np.sin(angles_o)
    inits.append(p3)
    
    # 4. Two concentric rings: 7 inner + 6 outer
    p4 = np.zeros((N, 2))
    angles_in = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p4[:7, 0] = 0.4 * np.cos(angles_in)
    p4[:7, 1] = 0.4 * np.sin(angles_in)
    angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p4[7:, 0] = np.cos(angles_out)
    p4[7:, 1] = np.sin(angles_out)
    inits.append(p4)
    
    # 5. Three layers: 1 center + 5 mid + 7 outer
    p5 = np.zeros((N, 2))
    angles_mid = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5[1:6, 0] = 0.45 * np.cos(angles_mid)
    p5[1:6, 1] = 0.45 * np.sin(angles_mid)
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
    p5[6:, 0] = np.cos(angles_out)
    p5[6:, 1] = np.sin(angles_out)
    inits.append(p5)
    
    # 6. Random
    rng = np.random.RandomState(42)
    p6 = rng.uniform(-1, 1, size=(N, 2))
    inits.append(p6)
    
    # 7. Another random
    rng2 = np.random.RandomState(123)
    p7 = rng2.uniform(-1, 1, size=(N, 2))
    inits.append(p7)
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses
    inits = _make_initial_guesses()
    for p in inits:
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = pts.copy()
    
    # 2. Multi-start local search: run local search from each initial guess
    #    and from the best found so far
    all_starts = inits.copy()
    if best_params is not None:
        all_starts.append(best_params)
    
    # Deduplicate
    unique_starts = []
    for s in all_starts:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)
    
    # Run local search from each start, with time budget
    num_starts = len(unique_starts)
    time_per_start = (DEADLINE - 5.0 - (time.time() - t_start)) / max(num_starts, 1)
    time_per_start = min(time_per_start, 8.0)  # Cap per-start time
    
    for start_idx, start in enumerate(unique_starts):
        remaining = DEADLINE - 5.0 - (time.time() - t_start)
        if remaining < 1.0:
            break
        
        budget = min(time_per_start, remaining)
        try:
            refined_pts, refined_score = _local_search(
                start, max_iter=300, time_budget=budget, t_start=time.time()
            )
            if refined_score > best_score:
                best_score = refined_score
                best_params = refined_pts.copy()
        except Exception:
            pass
    
    # 3. Final refinement: run local search on the best found
    remaining = DEADLINE - 5.0 - (time.time() - t_start)
    if remaining > 1.0 and best_params is not None:
        try:
            final_pts, final_score = _local_search(
                best_params, max_iter=500, time_budget=min(remaining, 10.0),
                t_start=time.time()
            )
            if final_score > best_score:
                best_score = final_score
                best_params = final_pts.copy()
        except Exception:
            pass
    
    # 4. One more round of local search with fresh random starts if time permits
    remaining = DEADLINE - 5.0 - (time.time() - t_start)
    if remaining > 3.0:
        rng = np.random.RandomState(999)
        for _ in range(3):
            if (time.time() - t_start) > (DEADLINE - 5.0):
                break
            rand_start = rng.uniform(-1, 1, size=(N, 2))
            # Mix with best: 50% random, 50% best
            mixed = 0.5 * rand_start + 0.5 * best_params
            try:
                refined_pts, refined_score = _local_search(
                    mixed, max_iter=200, time_budget=min(remaining, 4.0),
                    t_start=time.time()
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_params = refined_pts.copy()
            except Exception:
                pass
    
    return best_params.reshape(N, 2)
