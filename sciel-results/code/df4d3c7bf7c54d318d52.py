"""
Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Binary search on outer side length + Smooth SDF-based gradient descent.
Instead of minimizing size with penalties, we fix the size and try to find a valid
packing (no overlaps, all inside). We use differentiable Signed Distance Fields
(SDFs) for hexagons to compute smooth gradients for repulsion and containment,
allowing efficient navigation of the configuration space via L-BFGS-B.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0  # Safety margin
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_sdf(x, y, cx, cy, side, angle_deg):
    """
    Compute the signed distance from point (x, y) to a regular hexagon
    centered at (cx, cy) with side length 'side' and rotation 'angle_deg'.
    Positive if outside, negative if inside.
    
    This function is differentiable almost everywhere.
    """
    # Transform to local coordinates
    dx = x - cx
    dy = y - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    # Hexagon parameters in local frame (flat-top or pointy-top? 
    # Standard regular hexagon with vertices at 0, 60, ... degrees)
    # Inradius r = side * sqrt(3) / 2
    # Circumradius R = side
    # Let's use the standard definition where vertices are at angles 0, 60, ...
    # Wait, the _hex_vertices in the original code used:
    # a = angle_rad + k * pi / 3
    # So vertices are at 0, 60, 120... relative to angle.
    # This means the "flat" sides are perpendicular to the x-axis? No.
    # Vertex at 0 deg is (R, 0). Edge connects (R,0) to (R/2, R*sqrt(3)/2).
    # The normal to that edge is at 30 deg.
    # So the hexagon is "pointy" at 0, 60, etc.
    
    # Distance to hexagon can be computed by checking the 6 faces.
    # For a regular hexagon with circumradius R=side:
    # The distance from center to any face (inradius) is r = side * sqrt(3) / 2.
    # The hexagon is the intersection of 6 half-planes.
    # d = max(0, max_i (n_i . p - r))
    # Where n_i are unit normals to the faces.
    
    r = side * math.sqrt(3) / 2.0
    
    # Face normals are at 30, 90, 150, 210, 270, 330 degrees in local frame
    # (perpendicular to the vertices at 0, 60, ...)
    max_dist = -np.inf
    for k in range(6):
        a = math.radians(30 + k * 60)
        nx = math.cos(a)
        ny = math.sin(a)
        # Distance to the line of the face
        dist = lx * nx + ly * ny - r
        if dist > max_dist:
            max_dist = dist
            
    # If max_dist > 0, we are outside. The distance is max_dist.
    # If max_dist <= 0, we are inside. The distance is -r_in_to_center?
    # Actually, for SDF, if inside, dist should be negative.
    # The distance to the boundary from inside is r - (lx*nx + ly*ny) for the closest face?
    # No, the SDF of a convex polygon is:
    # if outside: max(0, max_i (n_i . p - h_i))
    # if inside: -min_i (h_i - n_i . p)
    
    # Let's re-evaluate.
    # h_i = r (inradius)
    # If p is inside, n_i . p < r for all i.
    # Closest face distance is min(r - n_i . p).
    # So SDF = -min(r - n_i . p) = max(n_i . p - r).
    # Wait, if p=0 (center), n_i . 0 = 0. max(0 - r) = -r. Correct.
    # If p is outside, say along normal 1. n_1 . p > r. max(n_1 . p - r) > 0. Correct.
    
    return max_dist

def _hex_sdf_grad(x, y, cx, cy, side, angle_deg):
    """
    Compute gradient of the hexagon SDF with respect to (x, y).
    """
    dx = x - cx
    dy = y - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    r = side * math.sqrt(3) / 2.0
    
    # Find the face that gives the max distance (closest face for outside, 
    # or the "limiting" face for inside)
    max_val = -np.inf
    best_nx = 0.0
    best_ny = 0.0
    
    for k in range(6):
        a = math.radians(30 + k * 60)
        nx = math.cos(a)
        ny = math.sin(a)
        val = lx * nx + ly * ny - r
        if val > max_val:
            max_val = val
            best_nx = nx
            best_ny = ny
            
    # Gradient in local frame is (best_nx, best_ny)
    # Transform back to global frame
    gx = best_nx * ca + best_ny * sa
    gy = -best_nx * sa + best_ny * ca
    
    return gx, gy

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """
    Calculate the minimum side length of the outer hexagon required to contain
    all inner hexagons.
    """
    cx, cy = outer_center
    best_side = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        # We need to check if the inner hexagon is inside the outer hexagon of side 's'.
        # Instead of iterating vertices, we can use the SDF of the outer hexagon.
        # But we want to find the minimal 's'.
        # For a fixed outer angle, the condition for a point (x,y) to be inside
        # a hexagon of side s centered at (cx,cy) is:
        # SDF_outer(x, y, cx, cy, s, outer_angle) <= 0
        # SDF_outer = max_k (n_k . (p-c) - s*sqrt(3)/2) <= 0
        # => s >= max_k (n_k . (p-c)) / (sqrt(3)/2)
        # => s >= 2/sqrt(3) * max_k (n_k . (p-c))
        
        # We need to satisfy this for ALL points in the inner hexagon.
        # The max of a convex function (like support function) over a convex set
        # occurs at the boundary. For a hexagon, we can just check the vertices.
        
        # Vertices of inner hexagon
        angle_rad = np.radians(iangle)
        for k in range(6):
            a = angle_rad + k * np.pi / 3
            vx = ix + 1.0 * np.cos(a)
            vy = iy + 1.0 * np.sin(a)
            
            dx = vx - cx
            dy = vy - cy
            o_angle_rad = np.radians(outer_angle)
            ca = np.cos(-o_angle_rad)
            sa = np.sin(-o_angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            # Max projection on face normals
            max_proj = -np.inf
            for j in range(6):
                a_j = math.radians(30 + j * 60)
                nx = math.cos(a_j)
                ny = math.sin(a_j)
                proj = lx * nx + ly * ny
                if proj > max_proj:
                    max_proj = proj
            
            required_side = 2.0 * max_proj / math.sqrt(3)
            if required_side > best_side:
                best_side = required_side
                
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """
    Check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon.
    2. No two inner hexagons overlap.
    """
    cx, cy = outer_center
    
    # Check containment
    for (ix, iy, iangle) in inner_positions:
        angle_rad = np.radians(iangle)
        for k in range(6):
            a = angle_rad + k * np.pi / 3
            vx = ix + 1.0 * np.cos(a)
            vy = iy + 1.0 * np.sin(a)
            
            # SDF of outer hexagon at this point
            d = _hex_sdf(vx, vy, cx, cy, outer_side, outer_angle)
            if d > tol:
                return False
                
    # Check overlaps using SDF
    # For two convex polygons, they are disjoint if the minimum distance between them > 0.
    # Approximation: Check if SDF of one at the vertices of the other is > 0?
    # No, that's not sufficient. 
    # Better: Use SAT or check if any vertex of A is inside B or vice versa?
    # For convex polygons, if they overlap, either a vertex of A is inside B, or a vertex of B is inside A, or edges intersect.
    # Since we are using SDF, we can check:
    # min_{p in A} SDF_B(p) > 0 OR min_{p in B} SDF_A(p) > 0
    # This is expensive. 
    # Given the smoothness, a simpler check for validity in the final step:
    # Use the SAT method from the original code for robustness, as it's discrete and reliable.
    
    def _hex_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * np.pi / 3
            verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
        return np.array(verts)
        
    def _hexagons_disjoint(verts1, verts2, tol=1e-7):
        def get_normals(verts):
            normals = []
            n = len(verts)
            for i in range(n):
                j = (i + 1) % n
                edge = verts[j] - verts[i]
                nx = -edge[1]
                ny = edge[0]
                norm = np.sqrt(nx*nx + ny*ny)
                if norm > 1e-12:
                    normals.append([nx/norm, ny/norm])
            return normals
        
        normals1 = get_normals(verts1)
        normals2 = get_normals(verts2)
        all_normals = normals1 + normals2
        
        for nx, ny in all_normals:
            proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
            proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - tol or max2 < min1 - tol:
                return True
        
        return False

    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
                
    return True

def _objective(x, outer_side, outer_angle):
    """
    Objective function: Sum of squared SDFs for overlaps and boundary violations.
    We want to drive this to 0.
    """
    positions = []
    for i in range(N):
        positions.append([x[i*3], x[i*3+1], x[i*3+2]])
        
    total_cost = 0.0
    
    # 1. Boundary penalty: Each vertex of each inner hexagon must be inside the outer hexagon.
    # SDF_outer > 0 means outside. We penalize max(0, SDF)^2.
    for (ix, iy, iangle) in positions:
        angle_rad = np.radians(iangle)
        for k in range(6):
            a = angle_rad + k * np.pi / 3
            vx = ix + 1.0 * np.cos(a)
            vy = iy + 1.0 * np.sin(a)
            
            d = _hex_sdf(vx, vy, 0.0, 0.0, outer_side, outer_angle)
            if d > 0:
                total_cost += d * d
                
    # 2. Overlap penalty: For each pair of inner hexagons, we want them to be disjoint.
    # Approximate by checking vertices of one against the other.
    # If a vertex of hex i is inside hex j (SDF < 0), we have a violation.
    # We penalize max(0, -SDF)^2.
    # This is O(N^2 * 6 * 6) SDF evaluations. N=11, so 11*10/2 * 36 = ~2000 evals.
    # Each eval is cheap.
    
    for i in range(N):
        ix, iy, iangle = positions[i]
        angle_rad_i = np.radians(iangle)
        
        for j in range(i+1, N):
            jx, jy, jangle = positions[j]
            
            # Check vertices of i against j
            for k in range(6):
                a = angle_rad_i + k * np.pi / 3
                vx = ix + 1.0 * np.cos(a)
                vy = iy + 1.0 * np.sin(a)
                
                d = _hex_sdf(vx, vy, jx, jy, 1.0, jangle)
                if d < 0:
                    total_cost += (-d) * (-d)
                    
            # Check vertices of j against i
            angle_rad_j = np.radians(jangle)
            for k in range(6):
                a = angle_rad_j + k * np.pi / 3
                vx = jx + 1.0 * np.cos(a)
                vy = jy + 1.0 * np.sin(a)
                
                d = _hex_sdf(vx, vy, ix, iy, 1.0, iangle)
                if d < 0:
                    total_cost += (-d) * (-d)
                    
    return total_cost

def _gradient(x, outer_side, outer_angle):
    """
    Compute gradient of the objective function.
    This is complex because the objective involves max/min operations.
    However, we can use numerical gradients or finite differences if analytical is too hard.
    Given the time constraint, let's try to use scipy's numerical gradient via L-BFGS-B 
    which doesn't require explicit gradient if not provided? 
    No, L-BFGS-B is faster with gradients. 
    But implementing the gradient of max(0, SDF)^2 requires chain rule and knowing which term is active.
    
    Alternative: Use BFGS or Nelder-Mead on the SDF objective?
    Nelder-Mead is robust but slow.
    Let's try to implement a simple numerical gradient or just use L-BFGS-B with approx_grad=True.
    """
    # For simplicity and robustness within the time limit, we will rely on L-BFGS-B's 
    # internal gradient approximation if we don't provide one, or use a simple finite difference.
    # However, L-BFGS-B with approx_grad can be slow.
    
    # Let's stick to the strategy: 
    # 1. Binary search on outer_side.
    # 2. For a fixed outer_side, run L-BFGS-B to minimize _objective.
    # 3. If _objective < tol, it's a valid packing.
    
    # We will NOT provide an analytical gradient to avoid bugs. We will let L-BFGS-B approximate it.
    # To speed this up, we might need to reduce the number of function evaluations.
    
    return None # Signal to use approx_grad

def _initial_lattice():
    """Create initial hexagonal lattice arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210, 270, 330]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def construct():
    """
    Main entry point.
    1. Binary search on outer side length.
    2. For each candidate side length, try to find a valid packing using L-BFGS-B.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Initial bounds for binary search
    # Lower bound: Theoretical minimum? 
    # Upper bound: A known valid configuration.
    
    # Let's start with a known valid configuration to get an upper bound.
    # The lattice configuration from _initial_lattice is likely valid for some side.
    init_pos = _initial_lattice()
    x0 = []
    for p in init_pos:
        x0.extend(p)
        
    # Calculate the side required for this initial config
    outer_angle = 0.0
    current_side = _min_enclosing_side(init_pos, (0.0, 0.0), outer_angle)
    
    # Check if it's valid (it should be, as it's a lattice)
    # But lattice might have overlaps if dist is too small.
    # dist = 2*sqrt(3) ~ 3.46. Min dist for non-overlap is 2*sqrt(3). So they touch.
    # So it is valid (disjoint).
    
    high = current_side
    low = 2.0 # Minimum possible side for 11 hexagons? 
    # 1 hexagon side 1. 11 hexagons area 11 * 3*sqrt(3)/2 ~ 28.6.
    # Outer hex area S^2 * 3*sqrt(3)/2.
    # S^2 >= 11 => S >= sqrt(11) ~ 3.3.
    # So low = 3.3 is a good start.
    low = 3.3
    
    best_result = None
    best_score = float('inf')
    
    # First, verify the initial config is valid and record it
    if _check_validity(init_pos, (0.0, 0.0), high, outer_angle, tol=1e-5):
        best_score = high
        best_result = (init_pos, (0.0, 0.0), high, outer_angle)
        
    # Binary search for the minimum valid side
    # We want to find the smallest S such that a valid packing exists.
    # We will try to optimize the packing for a given S.
    
    max_iterations = 50 # Limit iterations to stay within time
    iter_count = 0
    
    while high - low > 1e-4 and iter_count < max_iterations and _remaining_time() > 10.0:
        iter_count += 1
        mid = (low + high) / 2.0
        
        # Try to find a valid packing with outer side = mid
        # Use L-BFGS-B to minimize the violation cost
        
        # Use multiple starts
        found_valid = False
        best_cost = float('inf')
        best_x = None
        
        starts = [
            _initial_lattice(),
            _initial_random(),
            _initial_random()
        ]
        
        for start_pos in starts:
            if _remaining_time() < 5.0:
                break
                
            x0 = []
            for p in start_pos:
                x0.extend(p)
                
            try:
                res = minimize(
                    _objective, 
                    x0, 
                    args=(mid, outer_angle), 
                    method='L-BFGS-B', 
                    options={
                        'maxiter': 200, 
                        'maxfev': 500,
                        'ftol': 1e-6,
                        'gtol': 1e-5,
                        'approx_grad': True
                    }
                )
                
                cost = res.fun
                if cost < best_cost:
                    best_cost = cost
                    best_x = res.x
                    
                if cost < 1e-6:
                    found_valid = True
                    break
                    
            except Exception as e:
                continue
                
        if found_valid and best_x is not None:
            # Reconstruct positions
            positions = []
            for i in range(N):
                positions.append([best_x[i*3], best_x[i*3+1], best_x[i*3+2]])
                
            # Final validation
            if _check_validity(positions, (0.0, 0.0), mid, outer_angle, tol=1e-5):
                high = mid
                best_score = mid
                best_result = (positions, (0.0, 0.0), mid, outer_angle)
            else:
                # Optimization found a low cost but it's not strictly valid?
                # Maybe due to tolerance. Try to tighten or accept if cost is very low.
                if best_cost < 1e-4:
                    # It's "close enough" for the binary search step, but let's be safe.
                    # We'll treat it as invalid for the search bounds to be conservative.
                    pass
        else:
            low = mid
            
    # Return the best result found
    if best_result is not None:
        return best_result
    
    # Fallback
    return (init_pos, (0.0, 0.0), high, outer_angle)

if __name__ == "__main__":
    result = construct()
    print(f"Best side length: {result[2]}")
