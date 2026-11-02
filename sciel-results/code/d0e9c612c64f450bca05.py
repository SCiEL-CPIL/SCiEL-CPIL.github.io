"""Packing 12 unit hexagons in a regular hexagon using 6-fold symmetry and scipy optimization."""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution, basin_hopping, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint (max distance between any two points on unit hexagons)
        if dist_sq > 4.0:
            return True
        
        # Precise SAT check for all other cases
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Check all 12 edge normals (6 from each hexagon)
        for i in range(6):
            # Edge normal from h1
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            nn1 = np.linalg.norm(n1)
            if nn1 > 1e-12:
                n1 = n1 / nn1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
            
            # Edge normal from h2
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            nn2 = np.linalg.norm(n2)
            if nn2 > 1e-12:
                n2 = n2 / nn2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(len(inner))])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def objective_sym(params):
        """
        Objective with 6-fold symmetry.
        params: [r1, a1, r2, a2, r3, a3, outer_angle]
        where ri, ai define the position and rotation of hexagons in orbit i.
        Total hexagons: 6 * 3 = 18, but we only need 12.
        
        Actually, let's use a different approach. With 6-fold symmetry, we can have:
        - 1 hexagon at center (if it fits)
        - 6 hexagons in a ring at radius r1, angle a1
        - 5 hexagons in a ring at radius r2, angle a2 (breaking symmetry slightly)
        
        But that breaks strict 6-fold symmetry. Let me think differently.
        
        For 12 hexagons with 6-fold symmetry, we need orbits of size 6.
        So we can have:
        - 2 orbits of 6 hexagons each
        - Each orbit defined by (r, theta_offset, rotation)
        
        Total params: r1, theta1, rot1, r2, theta2, rot2, outer_angle = 7 params
        """
        if time.time() > deadline:
            return 1e6
        
        r1, theta1, rot1, r2, theta2, rot2, outer_angle = params
        
        # Generate all 12 hexagons
        inner = []
        # Orbit 1: 6 hexagons
        for k in range(6):
            angle_offset = np.radians(theta1) + k * np.pi / 3
            x = r1 * np.cos(angle_offset)
            y = r1 * np.sin(angle_offset)
            hex_rot = np.degrees(rot1) + k * 60.0
            inner.append((x, y, hex_rot))
        
        # Orbit 2: 6 hexagons
        for k in range(6):
            angle_offset = np.radians(theta2) + k * np.pi / 3
            x = r2 * np.cos(angle_offset)
            y = r2 * np.sin(angle_offset)
            hex_rot = np.degrees(rot2) + k * 60.0
            inner.append((x, y, hex_rot))
        
        center = (0.0, 0.0)
        
        # Check disjointness
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    penalty += 10.0
        
        if penalty > 0:
            return 1e4 + penalty
        
        # Compute side
        side = compute_min_side_fast(inner, center, outer_angle)
        
        return side

    def initial_guess():
        # Two rings of 6 hexagons
        # Inner ring: radius ~2, outer ring: radius ~4
        return np.array([2.0, 0.0, 0.0, 4.0, 30.0, 0.0, 0.0])

    # Use differential evolution for global search
    bounds = [
        (0.5, 6.0),   # r1
        (-30.0, 30.0), # theta1
        (-30.0, 30.0), # rot1
        (1.0, 8.0),   # r2
        (-30.0, 30.0), # theta2
        (-30.0, 30.0), # rot2
        (-30.0, 30.0), # outer_angle
    ]
    
    best_result = None
    best_score = float('inf')
    
    # Try differential evolution
    try:
        result = differential_evolution(
            objective_sym, bounds,
            maxiter=200, popsize=15,
            tol=1e-4, mutation=(0.5, 1.0),
            recombination=0.7, seed=42,
            polish=True, init='latinhypercube'
        )
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception as e:
        pass
    
    # Try basin hopping from best result or initial guess
    if best_result is None:
        best_result = initial_guess()
    
    def objective_bh(x):
        if time.time() > deadline:
            return 1e6
        return objective_sym(x)
    
    try:
        bh_result = basin_hopping(
            objective_bh, best_result,
            niter=50, T=0.1, stepsize=0.5,
            seed=42
        )
        if bh_result.fun < best_score:
            best_score = bh_result.fun
            best_result = bh_result.x
    except Exception as e:
        pass
    
    # Polish with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            nm_result = minimize(
                objective_bh, best_result,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_result = nm_result.x
        except Exception as e:
            pass
    
    # Extract best
    r1, theta1, rot1, r2, theta2, rot2, outer_angle = best_result
    
    inner = []
    for k in range(6):
        angle_offset = np.radians(theta1) + k * np.pi / 3
        x = r1 * np.cos(angle_offset)
        y = r1 * np.sin(angle_offset)
        hex_rot = np.degrees(rot1) + k * 60.0
        inner.append([float(x), float(y), float(hex_rot)])
    
    for k in range(6):
        angle_offset = np.radians(theta2) + k * np.pi / 3
        x = r2 * np.cos(angle_offset)
        y = r2 * np.sin(angle_offset)
        hex_rot = np.degrees(rot2) + k * 60.0
        inner.append([float(x), float(y), float(hex_rot)])
    
    center = (0.0, 0.0)
    side = compute_min_side_fast(inner, center, outer_angle)
    
    # Final validation
    valid = True
    for i in range(12):
        for j in range(i+1, 12):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use a known valid configuration (hexagonal grid)
        inner = []
        positions = [(0, 0, 0)]
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        for pos in positions:
            inner.append([float(pos[0]), float(pos[1]), float(pos[2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)
    
    return inner, center, side, float(outer_angle)
# EVOLVE-BLOCK-END
