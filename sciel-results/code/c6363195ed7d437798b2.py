import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    # Unit hexagon vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300)
    base_verts = np.array([(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)])
    
    def get_verts(cx, cy, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        return base_verts @ rot.T + np.array([cx, cy])

    def compute_min_dist(va, vb):
        """
        Compute the minimum distance between two convex polygons.
        If they overlap, return a negative value (penalty).
        If they are touching, return 0.
        If separated, return positive distance.
        """
        # Fast bounding box check
        min_xa, max_xa = np.min(va[:, 0]), np.max(va[:, 0])
        min_ya, max_ya = np.min(va[:, 1]), np.max(va[:, 1])
        min_xb, max_xb = np.min(vb[:, 0]), np.max(vb[:, 0])
        min_yb, max_yb = np.min(vb[:, 1]), np.max(vb[:, 1])
        
        if max_xa < min_xb or max_xb < min_xa or max_ya < min_yb or max_yb < min_ya:
            # Separated, return distance between boxes as a lower bound (or just large)
            dx = max(min_xb - max_xa, min_xa - max_xb, 0)
            dy = max(min_yb - max_ya, min_ya - max_yb, 0)
            return math.sqrt(dx*dx + dy*dy)

        # Check for overlap using SAT (Separating Axis Theorem)
        # For convex polygons, if there is a separating axis, they are disjoint.
        # If no separating axis is found, they overlap.
        
        # Get edges of both polygons
        edges_a = np.roll(va, -1, axis=0) - va
        edges_b = np.roll(vb, -1, axis=0) - vb
        
        # Normal vectors to edges
        # For edge (dx, dy), normal is (-dy, dx) or (dy, -dx)
        normals_a = np.column_stack([-edges_a[:, 1], edges_a[:, 0]])
        normals_b = np.column_stack([-edges_b[:, 1], edges_b[:, 0]])
        
        # Normalize normals
        norms_a = np.linalg.norm(normals_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(normals_b, axis=1, keepdims=True)
        # Avoid division by zero
        norms_a[norms_a == 0] = 1
        norms_b[norms_b == 0] = 1
        normals_a /= norms_a
        normals_b /= norms_b
        
        all_normals = np.vstack([normals_a, normals_b])
        
        # Project both polygons onto each normal
        # If projections do not overlap for any normal, polygons are disjoint.
        # We also need to track the minimum gap (distance) if disjoint.
        # If overlapping on all axes, we need to estimate the penetration depth (negative distance).
        
        min_gap = float('inf')
        is_disjoint = False
        
        for n in all_normals:
            # Project A
            proj_a = np.dot(va, n)
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            # Project B
            proj_b = np.dot(vb, n)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            
            # Check overlap
            if max_a <= min_b or max_b <= min_a:
                # Disjoint along this axis
                gap = max(min_b - max_a, min_a - max_b)
                if gap < min_gap:
                    min_gap = gap
                is_disjoint = True
                # We could break early if we find a separating axis, 
                # but for smooth optimization, we might want the "tightest" gap.
                # However, SAT says if ANY axis separates, they are disjoint.
                # The distance is the minimum gap over all separating axes? 
                # Actually, the distance is the minimum gap over all axes? No.
                # The distance is the minimum over all pairs of points.
                # For convex polygons, the minimum distance is the minimum gap over all separating axes?
                # Yes, for convex sets, the minimum distance is the minimum over all separating hyperplanes.
                # So we can break if we want, but calculating the min gap over all normals is safer for smoothness?
                # Actually, if we break, the function becomes non-smooth at the transition.
                # Let's compute the min gap over all normals to be smooth-ish.
                pass
        
        if is_disjoint:
            return min_gap
        
        # If we reach here, they overlap.
        # We need a penalty. A common proxy is the sum of squared penetration depths?
        # Or just a large negative number based on how deep the overlap is.
        # Simple approach: return a negative value proportional to the overlap area? Hard to compute.
        # Alternative: return a negative value based on the minimum "gap" (which would be negative if we extended the logic).
        # Let's define the "gap" on each axis. If min_a < min_b and max_a > min_b, gap is negative?
        # Let's re-evaluate.
        
        # Recalculate gaps for all normals
        min_gap = float('inf')
        for n in all_normals:
            proj_a = np.dot(va, n)
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            proj_b = np.dot(vb, n)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            
            # Overlap length on this axis
            overlap = min(max_a, max_b) - max(min_a, min_b)
            # If overlap > 0, they intersect on this axis.
            # If overlap < 0, they are separated by -overlap.
            if overlap < 0:
                gap = -overlap
                if gap < min_gap:
                    min_gap = gap
            
        # If min_gap is still inf, it means they overlap on ALL axes?
        # No, if they overlap on all axes, min_gap will be inf (since no axis had negative overlap).
        # Wait, if they overlap on all axes, they are intersecting.
        # In that case, we want a penalty.
        # Let's use the minimum overlap depth as a proxy for penetration.
        # Actually, if they intersect, the "distance" is negative.
        # Let's compute the maximum overlap depth?
        
        max_overlap = 0
        for n in all_normals:
            proj_a = np.dot(va, n)
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            proj_b = np.dot(vb, n)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap > max_overlap:
                max_overlap = overlap
                
        # Return negative overlap as penalty
        return -max_overlap

    def compute_outer_side(inner):
        """
        Compute the side length of the minimal enclosing regular hexagon.
        We assume the outer hexagon is centered at the centroid of the inner hexagons
        and aligned with the axes (angle 0). 
        Note: In a symmetric packing, this is likely optimal. 
        To be more robust, we could optimize the outer hexagon's angle and center too,
        but for now, let's fix the outer hexagon to be centered at origin and aligned.
        """
        # Centroid
        cx = np.mean([p[0] for p in inner])
        cy = np.mean([p[1] for p in inner])
        
        # Vertices of all inner hexagons
        all_verts = []
        for ix, iy, ia in inner:
            v = get_verts(ix - cx, iy - cy, ia) # Center them
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Minimal enclosing hexagon aligned with axes
        # The apothem of the outer hexagon S is the max projection of any point onto the 6 normal directions.
        # Normals are at 30, 90, 150, 210, 270, 330 degrees.
        apothem_factor = math.sqrt(3) / 2
        max_proj = 0
        for i in range(6):
            theta = math.radians(30 + 60*i)
            nx, ny = math.cos(theta), math.sin(theta)
            proj = np.dot(all_verts, np.array([nx, ny]))
            m = np.max(proj)
            if m > max_proj:
                max_proj = m
                
        S = max_proj / apothem_factor
        return S

    def evaluate_config(inner):
        """
        Objective function: Outer Side Length + Penalty for Overlaps.
        """
        S = compute_outer_side(inner)
        
        # Penalty for overlaps
        penalty = 0
        for i in range(N):
            va = get_verts(inner[i][0], inner[i][1], inner[i][2])
            for j in range(i+1, N):
                vb = get_verts(inner[j][0], inner[j][1], inner[j][2])
                d = compute_min_dist(va, vb)
                if d < 0:
                    # Quadratic penalty for overlap
                    penalty += (d * d) * 10.0
                    
        return S + penalty

    # Initialize with a "flower" pattern
    # 1 center, 6 around, 4 more?
    # Lattice distance for touching hexes is sqrt(3)
    dist = math.sqrt(3)
    
    # Center
    init = [[0.0, 0.0, 0.0]]
    # 6 neighbors
    for i in range(6):
        a = math.radians(60*i)
        init.append([dist * math.cos(a), dist * math.sin(a), 0.0])
    # 4 more? 
    # The 11 closest points to origin in hex lattice:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 neighbors
    # Next shell: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # We need 4 more. Let's pick 4 from the next shell to make it compact.
    # Or just use the 11 closest.
    # Axial to pixel: x = dist * (q + r/2), y = dist * (sqrt(3)/2 * r)
    # Closest 11:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2) ? 
    # Let's just generate the 11 closest and use that as init.
    
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = dist * (q + r/2.0)
            y = dist * (math.sqrt(3)/2.0 * r)
            lattice_points.append((x, y))
            
    sorted_points = sorted(lattice_points, key=lambda p: p[0]**2 + p[1]**2)
    init = [[p[0], p[1], 0.0] for p in sorted_points[:11]]
    
    # Center the init
    cx = sum(p[0] for p in init) / N
    cy = sum(p[1] for p in init) / N
    init = [[p[0]-cx, p[1]-cy, p[2]] for p in init]
    
    # Simulated Annealing
    current = [list(p) for p in init]
    current_score = evaluate_config(current)
    
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 1.0
    cooling = 0.999
    
    rng = random.Random(42)
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Perturb a random hexagon
        idx = rng.randrange(N)
        new_hex = list(current[idx])
        new_hex[0] += rng.gauss(0, 0.1)
        new_hex[1] += rng.gauss(0, 0.1)
        new_hex[2] += rng.gauss(0, 5) # Rotation perturbation
        
        # Ensure rotation is in [0, 60)
        new_hex[2] = new_hex[2] % 60
        
        new_config = [list(p) for p in current]
        new_config[idx] = new_hex
        
        new_score = evaluate_config(new_config)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < current_score:
            current = new_config
            current_score = new_score
            if current_score < best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current = new_config
                    current_score = new_score
        
        T *= cooling
        
    # Final local refinement with smaller steps
    T = 0.01
    for _ in range(1000):
        if time.time() > deadline:
            break
        idx = rng.randrange(N)
        new_hex = list(current[idx])
        new_hex[0] += rng.gauss(0, 0.01)
        new_hex[1] += rng.gauss(0, 0.01)
        new_hex[2] += rng.gauss(0, 0.5)
        new_hex[2] = new_hex[2] % 60
        
        new_config = [list(p) for p in current]
        new_config[idx] = new_hex
        new_score = evaluate_config(new_config)
        
        if new_score < current_score:
            current = new_config
            current_score = new_score
            if current_score < best_score:
                best = [list(p) for p in current]
                best_score = current_score
                
    # Verify best is valid (no overlaps)
    # The penalty should be 0 if valid.
    # If best_score is close to the side length, it's likely valid.
    # Let's double check with a hard check.
    
    def is_valid(config):
        for i in range(N):
            va = get_verts(config[i][0], config[i][1], config[i][2])
            for j in range(i+1, N):
                vb = get_verts(config[j][0], config[j][1], config[j][2])
                d = compute_min_dist(va, vb)
                if d < -1e-6: # Small tolerance
                    return False
        return True
        
    if is_valid(best):
        S = compute_outer_side(best)
        # Return format: inner, center, side, angle
        # inner: list of [x, y, angle_degrees]
        # center: (x, y) of outer hex
        # side: S
        # angle: 0.0 (aligned)
        
        # Center the inner hexagons at 0,0 for the return?
        # The interface says "outer hexagon center (x, y)".
        # If we return inner relative to (0,0), and outer center (0,0), it's consistent.
        
        return best, (0.0, 0.0), S, 0.0
    else:
        # Fallback to initial lattice config if SA failed to find valid
        S = compute_outer_side(init)
        return init, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
