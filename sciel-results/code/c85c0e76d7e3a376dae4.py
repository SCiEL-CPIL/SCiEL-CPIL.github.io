"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation for speed.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _generate_candidate(delta_angle):
    """
    Generate a candidate configuration based on a regular polygon with a specific angular perturbation.
    The points are placed at angles: theta_i = 2*pi*i/N + delta_angle * (some pattern)
    For simplicity, we start with the regular polygon and apply a small global rotation or specific perturbations.
    However, the known optimal structures often involve breaking the symmetry.
    
    A more robust approach: Start with regular polygon, then apply a specific perturbation pattern.
    Let's use a parameter 't' that controls a specific mode of deformation.
    For n=14, the optimal is likely close to the regular 14-gon but with some points moved.
    
    Actually, let's stick to the prompt's suggested strategy: log-spaced angular perturbation.
    We will generate points on a circle with angles:
    alpha_i = 2*pi*i/N + epsilon * sin(k * 2*pi*i/N)
    where k is a low frequency mode.
    
    Let's try a simpler heuristic: The optimal configuration for small n is often found by
    starting with a regular polygon and then performing a local search.
    To make it "materially different", we will use a deterministic grid search over a specific
    parameter space of angular deviations from the regular polygon.
    
    We parameterize the angles as:
    theta_i = 2*pi*i/N + delta * f(i)
    where f(i) is a fixed function (e.g., i - N/2) or similar.
    """
    angles_base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Try a few specific perturbation patterns
    # Pattern 1: Uniform rotation (doesn't change relative geometry, but helps with coordinate frame)
    # Pattern 2: Sinusoidal perturbation
    # theta_i = base_i + delta * sin(2*pi*m*i/N)
    
    # For this function, we just return the regular polygon for now, 
    # as the main search logic will handle the variations in construct().
    pts = np.column_stack([np.cos(angles_base), np.sin(angles_base)])
    return pts

def _refine_local(points, time_limit):
    """
    Perform local refinement using Powell's method on the Cartesian coordinates.
    """
    start_time = time.time()
    
    def objective(x):
        p = x.reshape(N, 2)
        # Simple penalty for non-convexity or self-intersection? 
        # Actually, we just want to maximize min triangle area / hull area.
        # If points become non-convex, the hull area might shrink or the min triangle area might drop.
        # The score function handles the normalization.
        s = _score(p)
        if s < 1e-9:
            return 1e10
        return -s

    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Powell',
            options={
                'maxiter': 500,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'maxfev': 1000
            }
        )
        if time.time() - start_time > time_limit:
            return points
        return res.x.reshape(N, 2)
    except Exception:
        return points

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Start with the regular 14-gon.
    2. Use a structured search over angular perturbations (low frequency modes).
    3. Refine the best candidates using local Cartesian optimization.
    4. Normalize the best result to unit area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Base regular polygon
    angles_base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_pts = np.column_stack([np.cos(angles_base), np.sin(angles_base)])
    
    best_score = _score(base_pts)
    best_pts = base_pts.copy()
    
    # Define a set of perturbation modes
    # We will perturb the angles: theta_i = base_i + delta * mode_i
    # Modes to try:
    # 1. Linear drift (breaks rotational symmetry slightly)
    # 2. Sine wave (k=1)
    # 3. Sine wave (k=2)
    # 4. Sine wave (k=3)
    
    modes = []
    # Mode 0: Identity (regular)
    modes.append(np.zeros(N))
    
    # Mode 1: Sin(2*pi*i/N)
    modes.append(np.sin(angles_base))
    
    # Mode 2: Sin(4*pi*i/N)
    modes.append(np.sin(2 * angles_base))
    
    # Mode 3: Sin(6*pi*i/N)
    modes.append(np.sin(3 * angles_base))
    
    # Mode 4: Cos(2*pi*i/N)
    modes.append(np.cos(angles_base))
    
    # Mode 5: Linear ramp (i/N - 0.5)
    modes.append(np.linspace(-0.5, 0.5, N, endpoint=False))
    
    # Search over delta values for each mode
    # Log-spaced deltas from small to large
    deltas = np.logspace(-3, 0, 20) # 0.001 to 1.0
    
    candidates = []
    
    for mode in modes:
        for delta in deltas:
            if time.time() > deadline - 30:
                break
            
            # Generate perturbed angles
            angles = angles_base + delta * mode
            
            # Ensure angles are sorted and within [0, 2pi)
            # Wrapping
            angles = np.mod(angles, 2 * np.pi)
            angles.sort()
            
            # Generate points
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Quick score evaluation
            s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
            
            # Keep top candidates for refinement
            # We can just keep a list of all decent ones
            if s > best_score * 0.95: # Keep ones that are close to best
                candidates.append(pts.copy())
        
        if time.time() > deadline - 30:
            break
    
    # Refine the top candidates
    # To save time, we will only refine the best few from the grid search
    # Sort candidates by score
    if candidates:
        # Evaluate scores for sorting
        scored_candidates = []
        for c in candidates:
            s = _score(c)
            scored_candidates.append((s, c))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Refine the top 3
        for i in range(min(3, len(scored_candidates))):
            if time.time() > deadline - 10:
                break
            
            s, c = scored_candidates[i]
            time_left = deadline - time.time() - 5
            if time_left < 5:
                break
            
            refined_pts = _refine_local(c, time_left)
            s_ref = _score(refined_pts)
            
            if s_ref > best_score:
                best_score = s_ref
                best_pts = refined_pts
    
    # Final check: if we haven't improved, just return the regular polygon (normalized)
    # But we should also try a few random restarts if time permits?
    # The prompt asks for a "materially different" strategy. The grid search over modes is different from pure Nelder-Mead.
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            final_points = best_pts * scale
        else:
            # Fallback
            final_points = base_pts
            hull = ConvexHull(final_points)
            hull_area = hull.volume
            scale = np.sqrt(1.0 / hull_area)
            final_points = final_points * scale
    except Exception:
        final_points = base_pts
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points

# EVOLVE-BLOCK-END
