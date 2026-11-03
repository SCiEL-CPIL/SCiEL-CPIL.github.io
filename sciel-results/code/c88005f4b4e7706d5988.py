"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Helper functions for hexagon geometry
    def hex_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon centered at (cx, cy) with side length and rotation."""
        angles = [math.radians(angle_deg + 60 * k) for k in range(6)]
        verts = []
        for a in angles:
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return verts
    
    def hex_support_function(theta):
        """
        Compute the support function of a unit regular hexagon in direction theta.
        For a unit regular hexagon (side length 1), the support function is:
        h(theta) = cos(pi/6) * max(|cos(theta - k*pi/3)|, |sin(theta - k*pi/3)|) for appropriate k
        Actually, for a regular hexagon with circumradius R = side (since side = R for regular hexagon):
        The support function is piecewise linear. For a unit hexagon, the support in direction (cos t, sin t) is:
        max over vertices v of v . (cos t, sin t)
        Since vertices are at angles 0, 60, 120, 180, 240, 300 degrees with radius 1:
        h(t) = max(cos(t), cos(t-60), cos(t-120), cos(t+120), cos(t+60), cos(t+180))
        = max over k of cos(t - k*pi/3)
        """
        t = theta
        vals = [math.cos(t - k * math.pi / 3) for k in range(6)]
        return max(vals)
    
    def outer_side_length_required(cx, cy, outer_center_x, outer_center_y, outer_angle_deg, inner_positions):
        """
        Compute the minimum side length of an outer hexagon centered at (outer_center_x, outer_center_y)
        with rotation outer_angle_deg that contains all inner hexagons.
        
        For two convex shapes A and B, B contains A translated by d if and only if
        for all directions u, h_B(u) >= h_A(u) + d . u
        where h is the support function.
        
        The outer hexagon must contain all inner hexagons. For a unit inner hexagon at position (x_i, y_i) with rotation a_i,
        and an outer hexagon of side length S centered at (cx, cy) with rotation c:
        The support of the outer hexagon in direction u is S * h_unit(u - c) where h_unit is the unit hexagon support.
        Wait, let me think more carefully.
        
        A regular hexagon with side length S has circumradius S. Its support function in direction theta is:
        h_S(theta) = S * max_k cos(theta - k*pi/3) (for a hexagon with a vertex on the positive x-axis)
        
        If the outer hexagon is rotated by angle c, its support is S * max_k cos(theta - c - k*pi/3)
        
        An inner hexagon of side 1 at center (x_i, y_i) with rotation a_i has support:
        h_i(theta) = max_k cos(theta - a_i - k*pi/3) + (x_i * cos(theta) + y_i * sin(theta)) - (cx * cos(theta) + cy * sin(theta))
        
        Wait, the support function of a shape shifted by vector d is h(theta) + d . (cos theta, sin theta).
        
        For the outer hexagon to contain the inner hexagon:
        S * max_k cos(theta - c - k*pi/3) >= h_inner_unit(theta - a_i) + (x_i - cx)*cos(theta) + (y_i - cy)*sin(theta)
        for all theta.
        
        So for a given outer center and angle, the minimum S needed is:
        S >= max over i, theta of [h_inner_unit(theta - a_i) + (x_i - cx)*cos(theta) + (y_i - cy)*sin(theta)] / max_k cos(theta - c - k*pi/3)
        
        Due to symmetry, we can restrict theta to a fundamental domain.
        """
        c = math.radians(outer_angle_deg)
        
        def outer_support(theta):
            vals = [math.cos(theta - c - k * math.pi / 3) for k in range(6)]
            return max(vals)
        
        min_S = 0.0
        # Sample theta densely
        num_samples = 360
        for i, (x_i, y_i, a_i) in enumerate(inner_positions):
            a_i_rad = math.radians(a_i)
            dx = x_i - cx
            dy = y_i - cy
            for j in range(num_samples):
                theta = 2 * math.pi * j / num_samples
                inner_support = hex_support_function(theta - a_i_rad) + dx * math.cos(theta) + dy * math.sin(theta)
                o_support = outer_support(theta)
                if o_support > 1e-10:
                    S_candidate = inner_support / o_support
                    if S_candidate > min_S:
                        min_S = S_candidate
        return min_S
    
    def total_side_length(cx, cy, outer_angle_deg, inner_positions):
        return outer_side_length_required(cx, cy, 0.0, 0.0, outer_angle_deg, inner_positions)
    
    # Generate candidate inner configurations
    def generate_candidates():
        candidates = []
        
        # Configuration 1: Linear arrangement
        N_linear = 5
        for offset in range(-3, 4):
            pos = [(2.0 * k, 0.0, 0.0) for k in range(-2, 3)]
            candidates.append(pos)
        
        # Configuration 2: Hexagonal packing (centered)
        # Place hexagons in a hexagonal lattice
        def hex_packing(n_rings):
            positions = [(0.0, 0.0, 0.0)]
            if n_rings >= 1:
                # Ring 1: 6 hexagons
                for k in range(6):
                    angle = math.radians(60 * k)
                    x = 2.0 * math.cos(angle)
                    y = 2.0 * math.sin(angle)
                    positions.append((x, y, 0.0))
            if n_rings >= 2:
                # Ring 2: 12 hexagons
                for k in range(6):
                    angle = math.radians(60 * k)
                    # Two hexagons in each sector
                    # Positions at distance 2*sqrt(3) and 4 along the direction
                    r1 = 2.0 * math.sqrt(3)
                    x1 = r1 * math.cos(angle)
                    y1 = r1 * math.sin(angle)
                    positions.append((x1, y1, 0.0))
                    r2 = 4.0
                    x2 = r2 * math.cos(angle)
                    y2 = r2 * math.sin(angle)
                    positions.append((x2, y2, 0.0))
                    # Additional positions
                    angle2 = angle + math.radians(30)
                    x3 = 4.0 * math.cos(angle2)
                    y3 = 4.0 * math.sin(angle2)
                    positions.append((x3, y3, 0.0))
                    angle3 = angle - math.radians(30)
                    x4 = 4.0 * math.cos(angle3)
                    y4 = 4.0 * math.sin(angle3)
                    positions.append((x4, y4, 0.0))
            return positions
        
        # Use up to 11 hexagons
        candidates.append(hex_packing(1)[:11])
        
        # Configuration 3: Compact cluster
        # Try to fit 11 hexagons in a compact arrangement
        # Center + 6 around + 4 more
        compact = [(0.0, 0.0, 0.0)]
        for k in range(6):
            angle = math.radians(60 * k)
            compact.append((2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.0))
        # Add 4 more in a tighter pattern
        for k in range(4):
            angle = math.radians(60 * k + 30)
            compact.append((3.0 * math.cos(angle), 3.0 * math.sin(angle), 0.0))
        candidates.append(compact)
        
        # Configuration 4: Two rows
        row1 = [(2.0 * k, 0.0, 0.0) for k in range(-2, 3)]
        row2 = [(2.0 * k + 1.0, math.sqrt(3), 0.0) for k in range(-2, 3)]
        candidates.append(row1 + row2[:1])  # 10 + 1 = 11
        
        # Configuration 5: Staggered
        staggered = []
        for k in range(-2, 3):
            staggered.append((2.0 * k, 0.0, 0.0))
        for k in range(-2, 3):
            staggered.append((2.0 * k + 1.0, math.sqrt(3), 0.0))
        candidates.append(staggered[:11])
        
        return candidates
    
    def optimize_center_and_angle(inner_positions):
        """Optimize the outer hexagon center and angle for a given inner configuration."""
        
        def objective(params):
            cx, cy, angle_deg = params
            return total_side_length(cx, cy, angle_deg, inner_positions)
        
        # Initial guess: center of mass, angle 0
        com_x = sum(p[0] for p in inner_positions) / len(inner_positions)
        com_y = sum(p[1] for p in inner_positions) / len(inner_positions)
        x0 = [com_x, com_y, 0.0]
        
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        return result.fun, result.x
    
    candidates = generate_candidates()
    best_score = float('inf')
    best_result = None
    best_config = None
    
    for idx, candidate in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Trim or pad to exactly 11 hexagons
        if len(candidate) > 11:
            candidate = candidate[:11]
        elif len(candidate) < 11:
            # Pad with hexagons far away (will be adjusted by optimization)
            while len(candidate) < 11:
                candidate.append((10.0, 10.0, 0.0))
        
        score, params = optimize_center_and_angle(candidate)
        
        if score < best_score:
            best_score = score
            best_result = params
            best_config = candidate
    
    if best_result is None:
        # Fallback to a simple configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    cx, cy, angle_deg = best_result
    
    # Convert to the expected format
    inner_list = [[x, y, a] for (x, y, a) in best_config]
    
    return inner_list, (cx, cy), best_score, angle_deg
# EVOLVE-BLOCK-END
