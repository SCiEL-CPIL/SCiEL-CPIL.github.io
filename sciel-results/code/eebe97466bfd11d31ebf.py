import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_log_ratio(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    """
    points = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize to fix scale: make the mean squared distance = 1
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max of log(squared distances)
    log_sq_dists = np.log(sq_dists)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Objective is the difference (log ratio)
    obj = soft_max_log - soft_min_log
    
    return obj

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t subject to constraints.
    Uses smooth approximation of constraints.
    x = [points_flat, t] where t is the log ratio.
    """
    n_params = N * D
    points = x[:n_params].reshape(N, D)
    t = x[n_params]
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize to fix scale: make the mean squared distance = 1
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq_dists = np.log(sq_dists)
    
    # Soft max of log(squared distances)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Penalty if soft_max - soft_min > t
    violation = soft_max_log - soft_min_log - t
    penalty = np.maximum(0, violation) ** 2
    
    return t + 100.0 * penalty

# New: Symmetry-reduced parameterization
def get_icosahedron_and_axis():
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Find a face center. 
    # Vertices 0, 1, 4 are adjacent?
    # 0: (0, 1, phi)
    # 1: (0, 1, -phi)
    # 4: (1, phi, 0)
    # Distance 0-1: 2phi
    # Distance 0-4: sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + 0.382 + 2.618) = sqrt(4) = 2.
    # Distance 1-4: sqrt(1 + (phi-1)^2 + phi^2) = 2.
    # So 0,1,4 form a face.
    v0, v1, v4 = ico[0], ico[1], ico[4]
    face_center = (v0 + v1 + v4) / 3.0
    axis = face_center / np.linalg.norm(face_center)
    
    return ico, axis

ICO_VERTS, ICO_AXIS = get_icosahedron_and_axis()

def generate_symmetric_points(params):
    """
    params: [r1, r2, theta]
    r1, r2: radial distances for the two extra points along the axis.
    theta: rotation angle of the icosahedron around the axis.
    """
    r1, r2, theta = params
    # Ensure positive radii for physical distance, but allow optimizer to explore
    # Actually, the points are at r1 * axis and r2 * (-axis).
    # If r is negative, it just flips direction.
    
    # Rotate icosahedron around ICO_AXIS by theta
    rot = Rotation.from_rotvec(theta * ICO_AXIS)
    ico_rot = rot.apply(ICO_VERTS)
    
    # Extra points
    p13 = r1 * ICO_AXIS
    p14 = r2 * (-ICO_AXIS)
    
    points = np.vstack([ico_rot, [p13, p14]])
    
    # Center the points
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize scale
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return None
    points = points / np.sqrt(mean_sq_dist)
    
    return points

def objective_symmetric(params):
    points = generate_symmetric_points(params)
    if points is None:
        return 1e10
    return raw_score(points)

def construct():
    start_time = time.time()
    timeout = 148.0  # Safety margin before 150s
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check for duplicate points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Phase 1: Symmetry-reduced search using Nelder-Mead
    # Parameters: r1, r2, theta
    # Initial guess: r1, r2 ~ distance of face center from origin
    # Face center distance: norm((v0+v1+v4)/3)
    face_dist = np.linalg.norm(ICO_AXIS) # This is 1.0 because we normalized the axis vector.
    # Wait, ICO_AXIS is normalized. The actual distance of the face center from origin is:
    # v0+v1+v4 has norm?
    # Let's just compute it.
    v0, v1, v4 = ICO_VERTS[0], ICO_VERTS[1], ICO_VERTS[4]
    fc = (v0 + v1 + v4) / 3.0
    r_face = np.linalg.norm(fc)
    
    # Search space for r1, r2: [0.5*r_face, 2.0*r_face]
    # Search space for theta: [0, 2*pi/5] (symmetry of icosahedron around face axis is 5-fold? No, 3-fold around face center)
    # The symmetry group of the icosahedron is A5. The stabilizer of a face is C3.
    # So theta in [0, 2*pi/3] is sufficient.
    
    nm_starts = 20
    for i in range(nm_starts):
        if time.time() - start_time > timeout - 30:
            break
        
        # Random initial guess for r1, r2, theta
        r1 = rng.uniform(0.5, 2.0) * r_face
        r2 = rng.uniform(0.5, 2.0) * r_face
        theta = rng.uniform(0, 2*np.pi/3)
        
        x0 = np.array([r1, r2, theta])
        
        try:
            res = minimize(
                objective_symmetric, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.x is not None:
                pts = generate_symmetric_points(res.x)
                update_best(pts)
                
                # If this is a good candidate, refine with full L-BFGS-B
                if pts is not None and raw_score(pts) < best_score * 1.1 and time.time() - start_time < timeout - 10:
                    x_full = pts.flatten()
                    try:
                        res2 = minimize(
                            objective_log_ratio, 
                            x_full, 
                            method='L-BFGS-B', 
                            args=(50.0,),
                            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                        )
                        if res2.x is not None:
                            pts2 = res2.x.reshape(N, D)
                            update_best(pts2)
                    except Exception:
                        pass
        except Exception:
            pass

    # Phase 2: Original multi-start search if time remains
    # (Kept from original code to ensure robustness if symmetric search fails)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Icosahedron + 2 points (various placements)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points at various symmetric positions
    for i in range(8):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        # Apply random rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 2. Fibonacci sphere with 14 points
    def fibonacci_sphere(n, rng):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        return pts
    
    for i in range(6):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())

    # 3. Random spherical points
    for i in range(6):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Uniform random in cube
    for i in range(4):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    n_restarts = len(candidates)
    
    # Tau schedule: start low for smooth landscape, increase for precision
    tau_schedule = [5.0, 15.0, 50.0, 150.0, 500.0, 1500.0]
    
    # Phase 3: Quick optimization with low tau on all candidates
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 4: Progressive tau increase with more iterations
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - start_time > timeout - 10:
            break
        tau = tau_schedule[tau_idx]
        max_iter = 100 + tau_idx * 40
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 5:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 5: High-precision polish on the best candidates
    if time.time() - start_time < timeout - 8:
        # Sort candidates by their current score
        scored_candidates = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored_candidates.append((score, i))
        scored_candidates.sort()
        
        # Polish the top 8 candidates with very high tau
        for score, i in scored_candidates[:8]:
            if time.time() - start_time > timeout - 4:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau_schedule[-1],),
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 6: Final exact local search using SLSQP on true objective
    if time.time() - start_time < timeout - 3:
        # Get top candidates
        scored_candidates = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored_candidates.append((score, i))
        scored_candidates.sort()
        
        def exact_objective(x):
            pts = x.reshape(N, D)
            sq_dists = get_pairwise_sq_dists(pts)
            min_sq = np.min(sq_dists)
            if min_sq < 1e-12:
                return 1e10
            return np.max(sq_dists) / min_sq
        
        for score, i in scored_candidates[:5]:
            if time.time() - start_time > timeout - 2:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    exact_objective, 
                    x0, 
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
