"""Hexagon packing: place N unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START

def _is_valid_solution(params, outer_side):
    """
    Check if a configuration of N hexagons is valid inside an outer hexagon of given side.
    params: array of shape (3*N,) -> [x0, y0, a0, x1, y1, a1, ...]
    Returns: True if valid, False otherwise.
    """
    positions = params.reshape(N, 3)
    
    # Precompute vertices for all inner hexagons
    # angles in radians
    angles_rad = np.deg2rad(positions[:, 2])
    
    # Unit hexagon vertices relative to center (angle 0)
    # Regular hexagon with circumradius 1
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.arange(6) * np.pi / 3.0
    base_vx = np.cos(base_angles)
    base_vy = np.sin(base_angles)
    
    # Rotate base vertices
    # cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
    # sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
    ca = np.cos(angles_rad)[:, np.newaxis]
    sa = np.sin(angles_rad)[:, np.newaxis]
    
    vx = positions[:, 0:1] + ca * base_vx - sa * base_vy
    vy = positions[:, 1:2] + sa * base_vx + ca * base_vy
    
    # Check pairwise disjointness
    # For each pair (i, j), check if any edge of i intersects edge of j or if any vertex is inside
    # A faster necessary condition: distance between centers > 2 * inradius? No, depends on rotation.
    # Robust check: 
    # 1. No vertex of i is inside j
    # 2. No edge of i intersects edge of j
    
    # To keep it fast, we can first check center distance.
    # Max width of hexagon is 2 (across corners). Min width is SQRT3 (across flats).
    # If center distance > 2 + 2 = 4, they definitely don't overlap.
    # If center distance < SQRT3/2 + SQRT3/2 = SQRT3 ~ 1.732, they definitely overlap? No, if rotated 30 deg, min dist is larger.
    # Actually, the minimum distance between two non-overlapping unit hexagons is 0 (touching).
    # The maximum distance between centers of two touching hexagons is 2 (flat-to-flat? No, corner-to-corner is 2+2=4? No.)
    # Let's stick to the geometric check but vectorize as much as possible.
    
    # Since N=11, N^2 is small. We can do a loop over pairs but vectorize the point-in-polygon and edge-intersection.
    
    for i in range(N):
        xi, yi, ai = positions[i]
        for j in range(i + 1, N):
            xj, yj, aj = positions[j]
            
            # Quick reject: center distance
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            # Max possible distance between centers of two touching hexagons is 2 (if flat-to-flat, dist is SQRT3? No.)
            # Circumradius is 1. 
            # If two hexagons touch flat-to-flat, distance is SQRT3.
            # If they touch corner-to-corner, distance is 2.
            # So if dist_sq > 4.0, they cannot overlap.
            if dist_sq > 4.0:
                continue
            
            # Get vertices
            vxi, vyi = vx[i], vy[i]
            vxj, vyj = vx[j], vy[j]
            
            # Check if any vertex of i is inside j
            # And if any vertex of j is inside i
            # And if edges intersect
            
            # Point in polygon (hexagon) check
            # A point (px, py) is inside hexagon centered at (cx, cy) with rotation a if:
            # Transform to local coords: (lx, ly)
            # |lx| <= 1, |ly| <= SQRT3/2, and |ly| <= SQRT3 * (1 - |lx|)
            
            def point_in_hex(px, py, cx, cy, angle_rad):
                dx_p = px - cx
                dy_p = py - cy
                # Rotate by -angle
                ca_p = np.cos(-angle_rad)
                sa_p = np.sin(-angle_rad)
                lx = ca_p * dx_p + sa_p * dy_p
                ly = -sa_p * dx_p + ca_p * dy_p
                
                if abs(lx) > 1.0: return False
                if abs(ly) > SQRT3 / 2.0: return False
                if abs(ly) > SQRT3 * (1.0 - abs(lx)): return False
                return True
            
            # Check vertices of i against j
            overlap = False
            for k in range(6):
                if point_in_hex(vxi[k], vyi[k], xj, yj, aj):
                    overlap = True
                    break
                if point_in_hex(vxj[k], vyj[k], xi, yi, ai):
                    overlap = True
                    break
            if overlap:
                return False
            
            # Check edge intersections
            # Edge i: (vxi[k], vyi[k]) to (vxi[k+1], vyi[k+1])
            # Edge j: (vxj[m], vyj[m]) to (vxj[m+1], vyj[m+1])
            
            def segments_intersect(p1, p2, p3, p4):
                # p1, p2, p3, p4 are (x, y) tuples
                def ccw(A, B, C):
                    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
                return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
            
            for k in range(6):
                p1 = (vxi[k], vyi[k])
                p2 = (vxi[(k+1)%6], vyi[(k+1)%6])
                for m in range(6):
                    p3 = (vxj[m], vyj[m])
                    p4 = (vxj[(m+1)%6], vyj[(m+1)%6])
                    if segments_intersect(p1, p2, p3, p4):
                        return False
                        
    # Check containment in outer hexagon
    # Outer hexagon: center (0,0), side L, angle 0 (flat top/bottom? No, standard orientation usually has flat top if angle 0 means vertex at 0? 
    # In the previous code: angles = a + arange(6)*pi/3. If a=0, vertex at (1,0). This is "pointy-top" relative to x-axis? 
    # Let's assume standard: vertices at 0, 60, 120... 
    # So "flat" sides are at 30, 90, 150...
    # The hexagon is defined by 6 half-planes.
    # For a hexagon with circumradius R (side length R) and rotation 0:
    # Vertices at (R, 0), (R/2, R*SQRT3/2), ...
    # The sides are:
    # 1. x + y/SQRT3 <= R  ? No.
    # Let's derive the inequalities for a hexagon with vertices at angles 0, 60, 120, 180, 240, 300 scaled by R.
    # Side 1 (between 0 and 60 deg): Line passes through (R, 0) and (R/2, R*SQRT3/2).
    # Slope: (R*SQRT3/2 - 0) / (R/2 - R) = (R*SQRT3/2) / (-R/2) = -SQRT3.
    # Eq: y - 0 = -SQRT3 (x - R) => SQRT3 x + y = R*SQRT3.
    # Inequality: SQRT3 x + y <= R*SQRT3.
    
    # Side 2 (between 60 and 120): Passes through (R/2, R*SQRT3/2) and (-R/2, R*SQRT3/2).
    # Horizontal line y = R*SQRT3/2.
    # Inequality: y <= R*SQRT3/2.
    
    # Side 3 (between 120 and 180): Passes through (-R/2, R*SQRT3/2) and (-R, 0).
    # Slope: (0 - R*SQRT3/2) / (-R - (-R/2)) = (-R*SQRT3/2) / (-R/2) = SQRT3.
    # Eq: y - 0 = SQRT3 (x + R) => -SQRT3 x + y = R*SQRT3.
    # Inequality: -SQRT3 x + y <= R*SQRT3.
    
    # By symmetry, the other 3 are:
    # -SQRT3 x - y <= R*SQRT3
    # -y <= R*SQRT3/2  => y >= -R*SQRT3/2
    # SQRT3 x - y <= R*SQRT3
    
    # So point (x,y) is inside if:
    # 1. SQRT3*x + y <= R*SQRT3
    # 2. y <= R*SQRT3/2
    # 3. -SQRT3*x + y <= R*SQRT3
    # 4. -SQRT3*x - y <= R*SQRT3
    # 5. -y <= R*SQRT3/2
    # 6. SQRT3*x - y <= R*SQRT3
    
    # We need ALL vertices of ALL inner hexagons to satisfy these.
    # Let's check the max values of these linear forms over all vertices.
    
    # All vertices X, Y (shape 6N)
    all_vx = vx.flatten()
    all_vy = vy.flatten()
    
    L = outer_side
    
    # Compute the 6 constraints for all points at once
    c1 = SQRT3 * all_vx + all_vy
    c2 = all_vy
    c3 = -SQRT3 * all_vx + all_vy
    c4 = -SQRT3 * all_vx - all_vy
    c5 = -all_vy
    c6 = SQRT3 * all_vx - all_vy
    
    if np.max(c1) > L * SQRT3 + 1e-9: return False
    if np.max(c2) > L * SQRT3 / 2.0 + 1e-9: return False
    if np.max(c3) > L * SQRT3 + 1e-9: return False
    if np.max(c4) > L * SQRT3 + 1e-9: return False
    if np.max(c5) > L * SQRT3 / 2.0 + 1e-9: return False
    if np.max(c6) > L * SQRT3 + 1e-9: return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Binary search on outer side L
    # Lower bound: Must contain at least one hexagon. Min L = 1.
    # Upper bound: Large enough to definitely fit. E.g., 20.
    low = 1.0
    high = 20.0
    
    best_valid_params = None
    best_L = high
    
    # We want to find the minimal L such that a valid configuration exists.
    # We can iterate L from low to high, or binary search.
    # Since the feasibility region is not necessarily convex, binary search on L might be tricky if there are "holes".
    # However, if it fits in L, it fits in L+epsilon. So the set of feasible L is [L_min, infinity).
    # So binary search works.
    
    # To speed up, we can run differential_evolution for a specific L.
    # If DE finds a solution, L is feasible.
    # If not, L might be infeasible (or DE failed).
    
    # Strategy:
    # 1. Try a few specific L values with DE.
    # 2. If found, tighten L.
    
    # Let's use a loop that tries to find a solution for a given L.
    def try_find_solution(L, max_iter=1000, popsize=10):
        # Bounds for DE:
        # x, y in [-L, L]
        # angle in [0, 60] (hexagon symmetry)
        bounds = []
        for i in range(N):
            bounds.append((-L, L)) # x
            bounds.append((-L, L)) # y
            bounds.append((0, 60)) # angle
        
        def objective(params):
            # We want to minimize the "violation" or just return 0 if valid, 1 if not.
            # DE minimizes. So we return 0 if valid, 1 if invalid.
            if _is_valid_solution(params, L):
                return 0.0
            else:
                return 1.0
        
        # Use DE
        try:
            res = differential_evolution(objective, bounds, maxiter=max_iter, popsize=popsize, 
                                          tol=0.01, seed=42, polish=False, workers=-1)
            if res.fun < 0.5:
                return res.x
        except Exception:
            pass
        return None

    # Initial guess for L
    # For N=11, a hexagonal arrangement is good.
    # Center + 6 + 12? No, 11 is 1+6+4? Or 1+6+... 
    # Hex numbers: 1, 7, 19.
    # 11 is between 7 and 19.
    # A cluster of 11 might fit in a hexagon of side ~ 3 or 4.
    
    # Let's try binary search steps
    # Step 1: Check if L=5 is feasible.
    # Step 2: If yes, try L=4, 3.5, 3.0...
    
    # To be safe and fast, we will try a sequence of L values.
    # If we find a valid one, we record it.
    # Then we try to go lower.
    
    L_candidates = [10.0, 8.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.2, 3.0, 2.8, 2.5]
    
    # Sort ascending? No, start with large to ensure we find *a* solution, then tighten.
    # Actually, starting small is better if we want to find the min quickly, but DE might fail if L is too small.
    # Let's start from a reasonable estimate and go down.
    
    # Estimate: 
    # Width of 11 hexagons in a row: 11*2 = 22? No, 10 gaps * 2 + 2 = 22.
    # Hexagon width = SQRT3 * L.
    # L >= 22 / SQRT3 ~ 12.7.
    # But we can stack them.
    # Hexagonal packing:
    # Row 1: 3
    # Row 2: 4
    # Row 3: 4
    # Total 11.
    # Width of row 4: 4*2 = 8? No, 3 gaps * 2 + 2 = 8.
    # Height: 3 rows. Row spacing = SQRT3.
    # Total height = 2*SQRT3 (for 3 rows? No, 2 gaps * SQRT3 + SQRT3 (height of hex) = 3*SQRT3? 
    # Let's say L ~ 4 or 5 is a good guess.
    
    # We will iterate through candidates from high to low.
    # If we find a valid config for L, we update best_L and best_params.
    # We stop if we run out of time.
    
    # To save time, we can reduce maxiter for smaller L if we are close to deadline.
    
    for L in L_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Adjust maxiter based on time remaining
        time_left = deadline - time.time()
        max_iter = int(min(200, time_left * 10)) # ~10 iterations per second? DE is slow.
        if max_iter < 10:
            max_iter = 10
            
        # Try to find solution for this L
        params = try_find_solution(L, max_iter=max_iter, popsize=8)
        
        if params is not None:
            # Verify again to be sure
            if _is_valid_solution(params, L):
                best_L = L
                best_valid_params = params
                # We found a solution, try to go lower in next iteration
                # But we are iterating fixed candidates. 
                # We could insert finer steps between best_L and the previous failed L.
                # For simplicity, let's just continue to the next lower candidate.
                
    if best_valid_params is not None:
        positions = best_valid_params.reshape(N, 3)
        inner = [[float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])] for i in range(N)]
        return inner, (0.0, 0.0), float(best_L), 0.0
    
    # Fallback: Return a valid but large solution
    # Place hexagons in a grid or row with large spacing
    inner = []
    # Simple row
    for i in range(N):
        inner.append([float(i * 2.5), 0.0, 0.0])
    # Estimate L for this row
    # Width = (N-1)*2.5 + 2 = 10*2.5 + 2 = 27
    # L = 27 / SQRT3 + 1 = 15.6 + 1 = 16.6
    L_fallback = 20.0
    return inner, (0.0, 0.0), L_fallback, 0.0

# EVOLVE-BLOCK-END
