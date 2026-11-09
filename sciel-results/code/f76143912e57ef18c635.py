"""
Strategy: Symmetry-Exploiting Hierarchical Optimization
Instead of generic SA/NM, we exploit the likely symmetry of the optimal packing for N=11.
We model the packing as a central cluster of 3 hexagons and two symmetric arms of 4 hexagons each.
We optimize the structural parameters (arm extension, rotation) using a deterministic
search with rigorous geometric validation (SAT) and exact outer hexagon calculation.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    SQRT3 = math.sqrt(3)
    
    # --- Geometry Helpers ---
    
    def get_hex_vertices(cx, cy, angle_deg):
        """Return 6 vertices of a unit hexagon centered at (cx, cy) rotated by angle_deg."""
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(angle_deg):
        """Return 6 unique normal axes for SAT."""
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        """SAT check for disjointness."""
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon centered at origin
        with rotation outer_angle_deg that contains all given hexagons.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # For a regular hexagon of side s, the distance from center to vertex is s.
        # The distance from center to edge (apothem) is s * sqrt(3)/2.
        # The projection of a point on the normal axis of the edge must be <= apothem.
        # So max_proj <= s * sqrt(3)/2  =>  s >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / SQRT3

    # --- Packing Configuration Generator ---
    
    def generate_packing(arm_len, arm_angle_deg, arm_twist_deg, central_twist_deg):
        """
        Generate a symmetric packing configuration.
        - Central cluster: 3 hexagons (one at center, two adjacent).
        - Arms: Two clusters of 4 hexagons, mirrored across x-axis.
        
        Parameters:
        - arm_len: Distance of the first arm hexagon from center.
        - arm_angle_deg: Angle of the first arm hexagon relative to x-axis.
        - arm_twist_deg: Rotation of the arm hexagons relative to the arm direction.
        - central_twist_deg: Rotation of the central cluster.
        """
        configs = []
        
        # Central Cluster:
        # Hex 0: Center (0,0)
        # Hex 1: (sqrt(3), 0) rotated by central_twist
        # Hex 2: (sqrt(3)/2, 1.5) rotated by central_twist
        # Note: These are touching if rotation is 0.
        
        c0 = (0.0, 0.0)
        c1 = (SQRT3, 0.0)
        c2 = (SQRT3/2, 1.5)
        
        configs.append((c0, central_twist_deg))
        configs.append((c1, central_twist_deg))
        configs.append((c2, central_twist_deg))
        
        # Arm 1 (Upper):
        # Start at (arm_len * cos(arm_angle), arm_len * sin(arm_angle))
        # Extend in a chain. To keep it compact, we use a "zig-zag" or straight line.
        # Let's try a straight line along the arm direction, with internal twists.
        
        base_angle_rad = math.radians(arm_angle_deg)
        dx = arm_len * math.cos(base_angle_rad)
        dy = arm_len * math.sin(base_angle_rad)
        
        # Arm hexagons positions relative to the start point
        # We place 4 hexagons in a row, separated by sqrt(3) if aligned, or less if twisted.
        # For simplicity and validity, we place them in a straight line along the arm direction.
        # The separation must be at least sqrt(3) if they are parallel.
        # If we twist them, we can pack tighter? No, twisting usually increases bounding box.
        # Let's assume the arm hexagons are parallel to each other and rotated by arm_twist_deg.
        
        sep = SQRT3 + 0.01 # Slight gap to ensure disjointness in initial gen, optimizer will close it
        
        for i in range(4):
            px = dx + i * sep * math.cos(base_angle_rad)
            py = dy + i * sep * math.sin(base_angle_rad)
            configs.append(((px, py), arm_angle_deg + arm_twist_deg))
            
        # Arm 2 (Lower): Mirror Arm 1 across X-axis
        for i in range(4):
            px = dx + i * sep * math.cos(-base_angle_rad)
            py = dy + i * sep * math.sin(-base_angle_rad)
            # Mirror the angle: if upper is angle + twist, lower is -angle - twist
            configs.append(((px, py), -arm_angle_deg - arm_twist_deg))
            
        return configs

    def evaluate_packing(configs, outer_angle_deg):
        """
        Validate disjointness and calculate outer side.
        Returns (is_valid, outer_side)
        """
        hexagons = []
        axes = []
        for (cx, cy), ang in configs:
            verts = get_hex_vertices(cx, cy, ang)
            hexagons.append(verts)
            axes.append(get_hex_axes(ang))
            
        # Check Disjointness
        for i in range(len(hexagons)):
            for j in range(i+1, len(hexagons)):
                if not is_disjoint(hexagons[i], hexagons[j], axes[i], axes[j]):
                    return False, float('inf')
                    
        # Calculate Outer Side
        side = calc_outer_side(hexagons, outer_angle_deg)
        return True, side

    # --- Optimization ---
    
    best_side = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    # Define search space for the symmetric parameters
    # arm_len: 2.0 to 4.0
    # arm_angle: 0 to 60 degrees (due to symmetry)
    # arm_twist: -30 to 30 degrees
    # central_twist: 0 to 60 degrees
    
    # Phase 1: Coarse Grid Search
    print("Phase 1: Coarse Search")
    grid_arm_len = np.arange(2.5, 4.5, 0.2)
    grid_arm_angle = np.arange(0, 60, 5)
    grid_arm_twist = np.arange(-30, 30, 10)
    grid_central_twist = np.arange(0, 60, 10)
    
    for al in grid_arm_len:
        if time.time() > deadline - 20: break
        for aa in grid_arm_angle:
            for at in grid_arm_twist:
                for ct in grid_central_twist:
                    configs = generate_packing(al, aa, at, ct)
                    # Try a few outer angles
                    for oa in [0, 10, 20, 30, 40, 50]:
                        valid, side = evaluate_packing(configs, oa)
                        if valid and side < best_side:
                            best_side = side
                            best_params = (al, aa, at, ct)
                            best_outer_angle = oa
                            
    print(f"Coarse best: {best_side} at {best_params}")
    
    # Phase 2: Local Refinement around best coarse point
    if best_params:
        print("Phase 2: Local Refinement")
        al0, aa0, at0, ct0 = best_params
        
        # Define objective for minimization
        def objective(p):
            al, aa, at, ct = p
            configs = generate_packing(al, aa, at, ct)
            # Use best_outer_angle as a starting guess, but maybe adjust slightly?
            # For now, fix outer angle to best found, or scan a small range.
            # Scanning outer angle inside objective is expensive.
            # Let's assume the outer angle is optimized separately or fixed near best.
            # Actually, the outer angle depends on the configuration.
            # Let's do a quick scan of 6 outer angles.
            min_side = float('inf')
            for oa in [best_outer_angle - 10, best_outer_angle, best_outer_angle + 10]:
                valid, side = evaluate_packing(configs, oa)
                if valid and side < min_side:
                    min_side = side
            if not math.isfinite(min_side):
                return 1000.0 # Penalty for invalid
            return min_side

        # Use Nelder-Mead on the 4 structural parameters
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_side:
                best_params = res.x
                # Re-evaluate to get exact best outer angle for this config
                al, aa, at, ct = best_params
                configs = generate_packing(al, aa, at, ct)
                for oa in np.arange(0, 60, 1):
                    valid, side = evaluate_packing(configs, oa)
                    if valid and side < best_side:
                        best_side = side
                        best_outer_angle = oa
        except Exception as e:
            print(f"Optimization error: {e}")
            
    # Phase 3: Final Fine-Tuning of Outer Angle and Validation
    print("Phase 3: Final Polish")
    al, aa, at, ct = best_params
    configs = generate_packing(al, aa, at, ct)
    
    # Fine scan outer angle
    final_side = float('inf')
    final_oa = 0.0
    for oa in np.arange(0, 60, 0.1):
        valid, side = evaluate_packing(configs, oa)
        if valid and side < final_side:
            final_side = side
            final_oa = oa
            
    # Final Validation and Centering
    # The packing is generated centered at (0,0) for the central hex.
    # We need to ensure the whole cluster is centered at (0,0) for the output.
    # The calc_outer_side assumes center is (0,0).
    # If the cluster is not centered at (0,0), the "min outer side" might be smaller
    # if we shift the cluster to center its bounding box.
    
    # Let's calculate the centroid of the inner hexagons and shift them.
    centers = [c for c, a in configs]
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    
    shifted_configs = [((c[0] - cx_avg, c[1] - cy_avg), a) for c, a in configs]
    
    # Re-evaluate with shifted configs
    hexagons = []
    axes = []
    for (cx, cy), ang in shifted_configs:
        verts = get_hex_vertices(cx, cy, ang)
        hexagons.append(verts)
        axes.append(get_hex_axes(ang))
        
    # Check validity again
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], axes[i], axes[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid configuration
        # Simple 3-5-3 lattice
        coords = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
        inner_list = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            inner_list.append([x, y, 0.0])
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        test_hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        test_axes = [get_hex_axes(p[2]) for p in inner_list]
        # Find best outer angle for fallback
        best_fb_side = float('inf')
        best_fb_oa = 0.0
        for oa in np.arange(0, 60, 1):
            side = calc_outer_side(test_hexagons, oa)
            if side < best_fb_side:
                best_fb_side = side
                best_fb_oa = oa
        return (inner_list, (0.0, 0.0), best_fb_side, best_fb_oa)

    # Find best outer angle for the shifted, valid configuration
    final_side = float('inf')
    final_oa = 0.0
    for oa in np.arange(0, 60, 0.1):
        side = calc_outer_side(hexagons, oa)
        if side < final_side:
            final_side = side
            final_oa = oa

    inner_list = []
    for (cx, cy), ang in shifted_configs:
        inner_list.append([cx, cy, ang])
        
    return (inner_list, (0.0, 0.0), final_side, final_oa)

# EVOLVE-BLOCK-END
