"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _initial_construction_regular():
    """14 points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _initial_construction_rings():
    """2 rings: 7 outer, 7 inner."""
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    r_outer = 1.0
    r_inner = 0.5
    
    outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Start with a symmetric, well-distributed initialization
    # Regular polygon is often a strong local optimum for min-area problems
    current_pts = _initial_construction_regular()
    current_score = _score(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    T0 = 0.05  # Initial temperature (perturbation magnitude)
    T_end = 1e-6
    cooling_rate = 0.9995
    
    T = T0
    
    # We will run as many steps as possible until deadline
    # Each step involves a score evaluation. 
    # Score evaluation is O(N^3) but vectorized. For N=14, it's fast.
    # Let's estimate steps: 120s / ~0.1ms per step = 1.2M steps? 
    # Actually, python loop overhead + numpy ops might be ~1-5ms per step.
    # Let's aim for ~10k-50k steps to be safe within time.
    
    steps = 0
    while time.time() < deadline:
        # Generate a new candidate by jittering one point
        idx = rng.integers(0, N)
        new_pts = current_pts.copy()
        
        # Jitter magnitude scales with temperature
        # We jitter in a small neighborhood
        jitter = rng.normal(0, T, size=(1, 2))
        new_pts[idx] += jitter
        
        # Optional: Keep points within a reasonable bound to avoid hull explosion
        # The score normalizes by hull area, so scaling doesn't matter much,
        # but extreme outliers can cause numerical issues or reduce min area ratio.
        # Let's clip to [-2, 2] just in case, though regular polygon is in [-1,1].
        # Actually, allowing expansion might help if the hull shrinks? 
        # No, we want to maximize min_area / hull_area.
        # If we expand hull, min_area likely increases too.
        # The ratio is scale-invariant.
        
        new_score = _score(new_pts)
        
        if new_score > current_score:
            # Always accept better
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept worse with probability exp(-delta / T)
            delta = current_score - new_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        steps += 1
        
        # Time check inside loop to avoid overshooting
        if time.time() > deadline:
            break

    # Final polish: If we have time, do a very fine-grained local search
    # on the best point found to squeeze out last bits of precision.
    if time.time() < deadline - 5.0:
        T_polish = 1e-4
        for _ in range(5000): # Fixed small number of polish steps
            if time.time() > deadline - 2.0:
                break
            idx = rng.integers(0, N)
            new_pts = best_pts.copy()
            jitter = rng.normal(0, T_polish, size=(1, 2))
            new_pts[idx] += jitter
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            T_polish *= 0.99

    return best_pts
# EVOLVE-BLOCK-END
