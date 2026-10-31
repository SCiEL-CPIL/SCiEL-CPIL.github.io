"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
# Equilateral triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
# Area = sqrt(3)/4
TRI_AREA = SQRT3 / 4.0

# Barycentric coordinates: point = b0*A + b1*B + b2*C, b0+b1+b2=1, bi>=0
# In Cartesian: x = b1 + 0.5*b2, y = (sqrt(3)/2)*b2
# So given (x,y): b2 = 2*y/sqrt(3), b1 = x - 0.5*b2, b0 = 1 - b1 - b2

def cartesian_to_barycentric(pts):
    """Convert (n,2) Cartesian to (n,3) barycentric."""
    x, y = pts[:, 0], pts[:, 1]
    b2 = 2.0 * y / SQRT3
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def barycentric_to_cartesian(bary):
    """Convert (n,3) barycentric to (n,2) Cartesian."""
    b0, b1, b2 = bary[:, 0], bary[:, 1], bary[:, 2]
    x = b1 + 0.5 * b2
    y = SQRT3 / 2.0 * b2
    return np.column_stack([x, y])

def min_triangle_area_bary(bary):
    """
    Compute the minimum triangle area (normalized by triangle area) 
    formed by any 3 of the N points, given in barycentric coordinates.
    
    The area of triangle formed by points with barycentric coords p_i, p_j, p_k
    (relative to the reference triangle) is |det([p_i; p_j; p_k])| / 2 * TRI_AREA.
    Normalized by TRI_AREA: |det| / 2.
    """
    n = bary.shape[0]
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        det = np.linalg.det(np.array([bary[i], bary[j], bary[k]]))
        area_norm = abs(det) / 2.0
        if area_norm < min_area:
            min_area = area_norm
    return min_area

def min_triangle_area_cartesian(pts):
    """Compute minimum triangle area normalized by triangle area."""
    n = pts.shape[0]
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        # Area = 0.5 * |cross(pj-pi, pk-pi)|
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        area_norm = area / TRI_AREA
        if area_norm < min_area:
            min_area = area_norm
    return min_area

def objective_bary(bary_flat):
    """
    Objective function for optimization in barycentric coordinates.
    Minimizes the negative of the minimum triangle area (to maximize it).
    We use a smooth approximation: soft-min of all triangle areas.
    """
    bary = bary_flat.reshape(-1, 3)
    # Enforce sum-to-1 and non-negativity via penalty
    # Actually, we'll use constraints instead
    n = bary.shape[0]
    # Compute all triangle areas
    areas = []
    for i, j, k in combinations(range(n), 3):
        det = np.linalg.det(np.array([bary[i], bary[j], bary[k]]))
        areas.append(abs(det) / 2.0)
    areas = np.array(areas)
    # Use soft-min: -log-sum-exp trick or just min for simplicity
    # For smoothness, use a power mean with low exponent
    # Actually, let's just return -min for the optimizer (it's piecewise smooth)
    return -np.min(areas)

def objective_bary_smooth(bary_flat, temperature=0.05):
    """
    Smooth objective using log-sum-exp approximation of min.
    min(x_i) ≈ -T * log(sum(exp(-x_i/T)))
    """
    bary = bary_flat.reshape(-1, 3)
    n = bary.shape[0]
    areas = []
    for i, j, k in combinations(range(n), 3):
        det = np.linalg.det(np.array([bary[i], bary[j], bary[k]]))
        areas.append(abs(det) / 2.0)
    areas = np.array(areas)
    # Soft min
    min_val = -temperature * np.log(np.sum(np.exp(-areas / temperature)))
    return -min_val  # We want to maximize min, so minimize -min

def make_constraints_bary(n):
    """Create constraints for barycentric coordinates."""
    # Each point: b0 + b1 + b2 = 1, b0 >= 0, b1 >= 0, b2 >= 0
    # We'll use bounds and equality constraints
    # Actually, for L-BFGS-B, we can use bounds and handle equality separately
    # Better: parameterize with only 2 coords per point (b1, b2), b0 = 1-b1-b2
    # Then bounds: b1 >= 0, b2 >= 0, b1+b2 <= 1
    
def objective_bary_reduced(b12_flat):
    """
    Objective using reduced parameters: for each point, only b1 and b2.
    b0 = 1 - b1 - b2 (must be >= 0, so b1+b2 <= 1)
    """
    n = N
    b12 = b12_flat.reshape(n, 2)
    b1, b2 = b12[:, 0], b12[:, 1]
    b0 = 1.0 - b1 - b2
    bary = np.column_stack([b0, b1, b2])
    
    # Check validity
    if np.any(b0 < -1e-10) or np.any(b1 < -1e-10) or np.any(b2 < -1e-10):
        return 1e10
    
    areas = []
    for i, j, k in combinations(range(n), 3):
        det = np.linalg.det(np.array([bary[i], bary[j], bary[k]]))
        areas.append(abs(det) / 2.0)
    areas = np.array(areas)
    return -np.min(areas)

def objective_bary_reduced_smooth(b12_flat, temperature=0.05):
    """Smooth version of the reduced objective."""
    n = N
    b12 = b12_flat.reshape(n, 2)
    b1, b2 = b12[:, 0], b12[:, 1]
    b0 = 1.0 - b1 - b2
    bary = np.column_stack([b0, b1, b2])
    
    if np.any(b0 < -1e-10) or np.any(b1 < -1e-10) or np.any(b2 < -1e-10):
        return 1e10
    
    areas = []
    for i, j, k in combinations(range(n), 3):
        det = np.linalg.det(np.array([bary[i], bary[j], bary[k]]))
        areas.append(abs(det) / 2.0)
    areas = np.array(areas)
    min_val = -temperature * np.log(np.sum(np.exp(-areas / temperature)))
    return -min_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Bounds for b1, b2: each in [0, 1], with b1+b2 <= 1
    # We'll use bounds [0,1] for each and rely on the objective penalty for b1+b2>1
    # Or better, use a transformation to ensure b1+b2 <= 1
    
    # Parameterize: for each point, use (u, v) in [0,1]^2
    # b1 = u * (1-v), b2 = v * (1-u), b0 = 1 - b1 - b2 = 1 - u(1-v) - v(1-u) = 1 - u - v + 2uv
    # Hmm, this doesn't quite work. Let me use a different approach.
    
    # Simpler: use b1, b2 directly with bounds [0, 1] and add penalty for b1+b2 > 1
    # Or use scipy's bounds with a custom constraint
    
    # Let's use the approach: optimize over (b1, b2) for each point with bounds
    # and a large penalty if b0 < 0
    
    bounds = [(0, 1)] * (2 * N)
    
    def penalized_objective(b12_flat):
        b12 = b12_flat.reshape(N, 2)
        b1, b2 = b12[:, 0], b12[:, 1]
        b0 = 1.0 - b1 - b2
        # Penalty for violating b0 >= 0
        penalty = 0.0
        if np.any(b0 < 0):
            penalty = 1e6 * np.sum(np.minimum(b0, 0)**2)
        
        bary = np.column_stack([b0, b1, b2])
        areas = []
        for i, j, k in combinations(range(N), 3):
            det = np.linalg.det(np.array([bary[i], bary[j], bary[k]]))
            areas.append(abs(det) / 2.0)
        areas = np.array(areas)
        return -np.min(areas) + penalty
    
    best_score = 0.0
    best_points = None
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    
    # Initial starts: random points in the triangle
    n_starts = 30
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initial barycentric coordinates
        b12_init = rng.random((N, 2))
        # Scale to ensure b1+b2 <= 1
        b12_init = b12_init / (1 + np.sum(b12_init, axis=1, keepdims=True))
        
        x0 = b12_init.flatten()
        
        try:
            result = minimize(
                penalized_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                b12_opt = result.x.reshape(N, 2)
                b1, b2 = b12_opt[:, 0], b12_opt[:, 1]
                b0 = 1.0 - b1 - b2
                
                # Check validity
                if np.all(b0 >= -1e-8) and np.all(b1 >= -1e-8) and np.all(b2 >= -1e-8):
                    bary = np.column_stack([np.maximum(b0, 0), np.maximum(b1, 0), np.maximum(b2, 0)])
                    # Normalize to ensure sum=1
                    bary = bary / np.sum(bary, axis=1, keepdims=True)
                    pts = barycentric_to_cartesian(bary)
                    
                    score = min_triangle_area_cartesian(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        
        except Exception:
            continue
    
    # Polish the best result with Nelder-Mead on the exact objective
    if best_points is not None:
        if time.time() < deadline - 20.0:
            try:
                bary_best = cartesian_to_barycentric(best_points)
                b12_best = np.column_stack([bary_best[:, 1], bary_best[:, 2]])
                
                # Nelder-Mead polish
                result_nm = minimize(
                    penalized_objective,
                    b12_best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                if result_nm.fun < 0:
                    b12_nm = result_nm.x.reshape(N, 2)
                    b1, b2 = b12_nm[:, 0], b12_nm[:, 1]
                    b0 = 1.0 - b1 - b2
                    
                    if np.all(b0 >= -1e-8) and np.all(b1 >= -1e-8) and np.all(b2 >= -1e-8):
                        bary = np.column_stack([np.maximum(b0, 0), np.maximum(b1, 0), np.maximum(b2, 0)])
                        bary = bary / np.sum(bary, axis=1, keepdims=True)
                        pts = barycentric_to_cartesian(bary)
                        
                        score = min_triangle_area_cartesian(pts)
                        if score > best_score:
                            best_score = score
                            best_points = pts
            except Exception:
                pass
    
    # If no valid result, return a default
    if best_points is None:
        # Fallback: grid-like placement
        pts = []
        for i in range(N):
            # Place points in a rough grid inside the triangle
            row = i // 4
            col = i % 4
            # Barycentric coordinates for a grid
            b2 = 0.8 * (row + 0.5) / 4
            b1 = 0.8 * (col + 0.5) / 4 * (1 - b2)
            b0 = 1 - b1 - b2
            pts.append([b1 + 0.5 * b2, SQRT3 / 2 * b2])
        best_points = np.array(pts)
    
    return best_points
