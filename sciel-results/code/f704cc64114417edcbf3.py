import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 14

# Precompute triple indices for N=14
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 140s to be safe.
    deadline = start_time + 140.0

    # Objective function for minimizers (maximize score -> minimize -score)
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -get_score(pts)

    best_points = None
    best_score = 0.0

    rng = np.random.default_rng(42)

    # 1. Generate diverse initializations
    initializations = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    reg_poly = np.column_stack((x, y))
    initializations.append(reg_poly)
    
    # Perturbed Regular Polygons
    # Small perturbations to break symmetry.
    for i in range(50):
        # Random rotation
        theta = rng.uniform(0, 2*np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts = reg_poly @ R.T
        
        # Random radial perturbation
        # Keep points mostly on the hull, but allow some inward movement
        # If we move inward, hull area shrinks. 
        # Let's try uniform radial perturbation in [0.9, 1.0]
        radii = rng.uniform(0.8, 1.0, N)
        pts = pts * radii[:, np.newaxis]
        
        # Small angular perturbation
        ang_noise = rng.normal(0, 0.05, N)
        new_angles = angles + ang_noise
        # Sort angles to maintain convexity order? No, just use the points.
        # Actually, if we perturb angles and radii, we might lose convexity.
        # Let's just use the points as is. If they are not convex, ConvexHull will handle it.
        # But we want them to be in convex position ideally.
        
        initializations.append(pts)

    # Random points in unit disk, filtered for "goodness"
    # We want points that are well-separated.
    for i in range(100):
        # Sample random points
        pts = rng.uniform(-1, 1, (N, 2))
        # Check if they form a convex hull with decent area
        try:
            hull = ConvexHull(pts)
            if hull.volume > 0.5: # Arbitrary threshold
                initializations.append(pts)
        except:
            pass

    # 2. Ring-based topologies (Center + Outer, 3+11, 4+10, etc.)
    # These often outperform flat polygons by allowing better distribution of area.
    # We parameterize rings by angles and radii.
    
    def ring_objective(params, topology):
        """
        params: flat array of angles and radii for the rings.
        topology: list of (count, base_radius) tuples? Or just counts?
        Let's define topology as a list of ring counts, e.g., [7, 7] or [1, 13] or [3, 11].
        The radii will be optimized as part of the parameters if we want, 
        but to keep dimensionality low, let's fix relative radii or optimize them.
        
        For simplicity in this initial generation, let's just generate specific ring configs
        and feed them into the general optimizer.
        """
        pass # We will generate specific points instead

    def generate_ring_points(counts, rng):
        """
        Generate points for a given ring topology (list of counts per ring).
        Radii are chosen to be spaced out, angles are randomized.
        """
        pts = []
        total_points = sum(counts)
        if total_points != N:
            return None
            
        # Determine radii. We want to maximize hull area, so outer ring should be large.
        # Inner rings smaller.
        # Let's try logarithmic spacing or linear spacing.
        # Max radius 1.0.
        num_rings = len(counts)
        if num_rings == 1:
            radii = [1.0]
        else:
            # Outer ring at 1.0, inner rings scaled down
            # Simple heuristic: r_k = 1.0 * (k + 1) / num_rings
            # But center point (count=1) should be at 0?
            # If count is 1, it's likely a center point.
            radii = []
            for i, count in enumerate(counts):
                if count == 1 and i == 0:
                    radii.append(0.0)
                else:
                    # Distribute radii between 0.3 and 1.0
                    # Or just use 1.0 for outer, and scale inner
                    pass
            
            # Let's just use specific radii for known good topologies
            if counts == [1, 13]:
                radii = [0.0, 1.0]
            elif counts == [3, 11]:
                radii = [0.5, 1.0]
            elif counts == [4, 10]:
                radii = [0.6, 1.0]
            elif counts == [5, 9]:
                radii = [0.6, 1.0]
            elif counts == [6, 8]:
                radii = [0.7, 1.0]
            elif counts == [7, 7]:
                radii = [0.8, 1.0]
            else:
                # Fallback
                radii = [1.0 * (i+1)/num_rings for i in range(num_rings)]
                
        for ring_idx, count in enumerate(counts):
            r = radii[ring_idx]
            if count == 1 and r < 0.1:
                # Center point
                pts.append([0.0, 0.0])
            else:
                # Random angles for this ring
                # Ensure good spacing by using linspace + noise
                base_angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                noise = rng.normal(0, 0.1, count)
                ang = base_angles + noise
                x = r * np.cos(ang)
                y = r * np.sin(ang)
                ring_pts = np.column_stack((x, y))
                pts.append(ring_pts)
                
        if not pts:
            return None
            
        return np.vstack(pts)

    # Generate ring initializations
    ring_topologies = [
        [1, 13],
        [3, 11],
        [4, 10],
        [5, 9],
        [6, 8],
        [7, 7],
        [2, 12],
        [1, 5, 8],
        [1, 6, 7],
        [1, 4, 9]
    ]
    
    for topo in ring_topologies:
        for _ in range(10): # 10 randomizations per topology
            pts = generate_ring_points(topo, rng)
            if pts is not None:
                initializations.append(pts)

    # 3. Optimize each initialization
    for init_pts in initializations:
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Powell's method
            res = minimize(
                objective, 
                init_pts.flatten(), 
                method='Powell', 
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 50000}
            )
            
            if res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception:
            pass

    # 4. Final fine-tuning with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            res_nm = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 100000}
            )
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        best_points = np.column_stack((x, y))

    return best_points
