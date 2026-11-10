import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D using Nelder-Mead optimization 
    from multiple symmetric initializations.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620  # Leave 10s margin
    
    def get_score(pts):
        """
        Calculates the metric (d_min / d_max)^2.
        Returns 0 if invalid.
        """
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        if d_max_sq < 1e-12:
            return 0.0
        
        # Metric is (d_min / d_max)^2 = d_min_sq / d_max_sq
        return d_min_sq / d_max_sq

    def get_negative_score(pts):
        return -get_score(pts)

    def objective(flat_vec):
        # Reconstruct points from flattened vector
        # Structure of flat_vec:
        # Index 0: x1 (fixed to 0 in optimization, but we can just ignore it or set it)
        # Actually, to save dimensions:
        # We fix P0 = (0,0,0)
        # We fix P1 = (r, 0, 0) where r is the first variable.
        # The remaining 12 points have 3 coordinates each.
        
        # Let's define the variable vector explicitly for clarity in reconstruction.
        # Total variables: 1 (for P1_x) + 12 * 3 (for P2..P13) = 37 variables.
        
        r = flat_vec[0]
        remaining = flat_vec[1:]
        
        # Ensure r is positive to avoid symmetry issues with P1 being at -x
        # Although Nelder-Mead handles signs, fixing P1 on +x axis reduces search space.
        if r < 0:
            r = -r
            
        pts = np.zeros((n, 3))
        pts[0] = [0, 0, 0]
        pts[1] = [r, 0, 0]
        
        # Fill the rest
        # remaining has size 36 (12 points * 3 coords)
        for i in range(12):
            start_idx = i * 3
            pts[i + 2] = remaining[start_idx:start_idx + 3]
            
        return get_negative_score(pts)

    # Initialize a good symmetric seed
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    # We need to place 14 points.
    # Let's try to place 12 points on a sphere of radius R, and 2 points near the center or on the sphere.
    # A common high-symmetry configuration for 14 points in 3D might be related to the icosahedron plus 2 points.
    # However, simply adding 2 points to an icosahedron might not be optimal.
    # Let's generate multiple diverse initial candidates and pick the best one after a quick local polish.
    
    candidates = []
    
    # Candidate 1: Icosahedron + 2 points at origin? No, coincident points are bad.
    # Let's try Icosahedron (scaled) + 2 points slightly off-center.
    # Or, a different polytope.
    
    # Let's just use a few random but structured starts.
    # Start 1: Icosahedron vertices on unit sphere, plus 2 points on unit sphere at random positions.
    p1 = ico_verts.copy()
    extra1 = np.array([1, 0, 0])
    extra2 = np.array([0, 1, 0])
    p1 = np.vstack([p1, [extra1], [extra2]])
    # Center it
    p1 = p1 - np.mean(p1, axis=0)
    candidates.append(p1)
    
    # Start 2: Random Gaussian
    np.random.seed(42)
    p2 = np.random.randn(14, 3)
    p2 = p2 - np.mean(p2, axis=0)
    candidates.append(p2)
    
    # Start 3: Random Uniform in Cube
    p3 = np.random.rand(14, 3) * 2 - 1
    p3 = p3 - np.mean(p3, axis=0)
    candidates.append(p3)
    
    # Start 4: Icosahedron + 2 points near center (small radius)
    p4 = ico_verts.copy()
    extra1 = np.array([0.1, 0, 0])
    extra2 = np.array([0, 0.1, 0])
    p4 = np.vstack([p4, [extra1], [extra2]])
    p4 = p4 - np.mean(p4, axis=0)
    candidates.append(p4)
    
    # Pre-process candidates to fit the parameterization
    # We need to convert each candidate into the 37-dim vector format.
    # 1. Translate so P0 is at origin? No, the optimization handles translation via P0 fixed at 0.
    #    So we just take the coordinates relative to P0.
    #    But P0 is arbitrary. Let's pick the point with smallest norm as P0?
    #    Or just take the first point as P0, translate all points by -P0.
    #    Then rotate so that P1 (second point) is on the +X axis.
    
    def to_param_space(pts):
        # pts is (14, 3)
        # 1. Translate so first point is at origin
        p0 = pts[0].copy()
        pts_trans = pts - p0
        
        # 2. Identify the point farthest from p0? Or just take pts[1]?
        #    The parameterization fixes P1 on X-axis.
        #    Let's pick the point with the largest distance from P0 as P1 to maximize the scale 'r'.
        dists_to_p0 = np.linalg.norm(pts_trans, axis=1)
        dists_to_p0[0] = 0
        max_idx = np.argmax(dists_to_p0)
        
        # Swap pts[1] and pts[max_idx] if max_idx != 1
        if max_idx != 1:
            pts_trans[1], pts_trans[max_idx] = pts_trans[max_idx].copy(), pts_trans[1].copy()
            
        # Now pts_trans[0] is 0, pts_trans[1] is the farthest point.
        
        # 3. Rotate so pts_trans[1] is on +X axis.
        p1 = pts_trans[1]
        r = np.linalg.norm(p1)
        if r < 1e-9:
            # Degenerate case, add small noise
            return np.zeros(37)
        
        # Rotation matrix to align p1 with [1, 0, 0]
        # We can use scipy or manual calculation.
        # Let's use a simple method:
        # u = p1 / r
        # We want R such that R @ u = [1, 0, 0]
        
        # Construct rotation from vector u to x-axis
        # This is a standard problem.
        # If u is close to x-axis, rotation is small.
        
        # Using Rodrigues' formula or similar.
        # Let's just use scipy.spatial.transform.Rotation if available?
        # Or simpler:
        # x = [1, 0, 0]
        # v = u
        # angle = arccos(dot(x, v))
        # axis = cross(x, v)
        
        x_axis = np.array([1.0, 0.0, 0.0])
        u = p1 / r
        
        dot_prod = np.dot(x_axis, u)
        if dot_prod > 0.999999:
            # Already aligned
            rot_pts = pts_trans
        elif dot_prod < -0.999999:
            # Anti-aligned, rotate 180 around Y
            rot_pts = pts_trans.copy()
            rot_pts[:, 0] = -rot_pts[:, 0]
            rot_pts[:, 2] = -rot_pts[:, 2]
        else:
            axis = np.cross(x_axis, u)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-9:
                rot_pts = pts_trans
            else:
                axis = axis / axis_norm
                angle = np.arccos(dot_prod)
                
                # Rotation matrix from axis-angle
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                
                rot_pts = (R @ pts_trans.T).T
        
        # Now rot_pts[0] should be 0, rot_pts[1] should be [r, 0, 0] (approx)
        
        # Extract parameters
        param_vec = np.zeros(37)
        param_vec[0] = r
        
        # Fill remaining 12 points (indices 2 to 13)
        # Note: The order of the remaining 12 points doesn't matter for the score, 
        # but it matters for the Nelder-Mead simplex. 
        # We just dump them in order.
        for i in range(12):
            idx = i + 2
            param_vec[1 + i*3 : 1 + i*3 + 3] = rot_pts[idx]
            
        return param_vec

    # Evaluate candidates in parameter space to find the best starting point
    best_start_vec = None
    best_start_score = -np.inf
    
    for cand in candidates:
        vec = to_param_space(cand)
        score = -objective(vec)
        if score > best_start_score:
            best_start_score = score
            best_start_vec = vec
            
    # Run Nelder-Mead optimization
    # We have plenty of time (620s). Nelder-Mead is fast.
    # We can run it multiple times with different perturbations if time permits.
    
    best_vec = best_start_vec
    best_score = best_start_score
    
    # Options for Nelder-Mead
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'disp': False
    }
    
    # Perform optimization
    # To be safe and potentially find better local minima, we can run from multiple starts
    # or restart if we get stuck.
    
    # Let's run a single high-quality optimization from the best candidate.
    # If we have time, we can run more.
    
    res = minimize(objective, best_start_vec, method='Nelder-Mead', options=options)
    
    if res.success or (not res.success and res.fun < best_score):
        # Check if new score is better
        new_score = -res.fun
        if new_score > best_score:
            best_score = new_score
            best_vec = res.x
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        # Try a second run with a slightly perturbed start or different method?
        # Or just refine the current one more?
        # Let's try running Nelder-Mead again from the current best, with tighter tolerances?
        # Or maybe a different initial guess?
        
        # Let's try a few random restarts if time allows.
        num_restarts = 5
        for i in range(num_restarts):
            if time.time() > deadline - 60:
                break
                
            # Perturb the current best vector
            noise = np.random.randn(37) * 0.1
            start_vec = best_vec + noise
            
            res = minimize(objective, start_vec, method='Nelder-Mead', options=options)
            new_score = -res.fun
            if new_score > best_score:
                best_score = new_score
                best_vec = res.x
                
    # Reconstruct points from best_vec
    r = best_vec[0]
    if r < 0: r = -r
    remaining = best_vec[1:]
    
    pts = np.zeros((n, 3))
    pts[0] = [0, 0, 0]
    pts[1] = [r, 0, 0]
    
    for i in range(12):
        start_idx = i * 3
        pts[i + 2] = remaining[start_idx:start_idx + 3]
        
    # Final validation
    if not np.all(np.isfinite(pts)):
        # Fallback to a simple valid configuration if something went wrong
        pts = np.random.randn(14, 3)
        
    return pts
