"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(center, angle_deg, side=1.0):
        cx, cy = center
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(center, angle_deg):
        """Returns the 6 outward normal vectors for the hexagon edges."""
        angle_rad = math.radians(angle_deg)
        axes = []
        # Normals are at 30 + 60k degrees relative to the hexagon orientation
        # Wait, vertices are at 0, 60...
        # Edges are between vertices.
        # Normal to edge between v0 and v1 is at 30 degrees.
        for k in range(6):
            theta = angle_rad + math.radians(30 + 60 * k)
            axes.append((math.cos(theta), math.sin(theta)))
        return axes

    def polygon_overlap_depth(verts1, axes1, verts2, axes2):
        """
        Calculate the minimum penetration depth between two convex polygons.
        Returns 0 if disjoint, >0 if overlapping.
        """
        min_depth = float('inf')
        
        all_axes = axes1 + axes2
        
        for ax in all_axes:
            # Project verts1
            proj1 = [v[0]*ax[0] + v[1]*ax[1] for v in verts1]
            min1, max1 = min(proj1), max(proj1)
            
            # Project verts2
            proj2 = [v[0]*ax[0] + v[1]*ax[1] for v in verts2]
            min2, max2 = min(proj2), max(proj2)
            
            # Calculate overlap
            # If max1 < min2 or max2 < min1, no overlap on this axis
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Overlap length
            # The overlap is the intersection of [min1, max1] and [min2, max2]
            overlap_start = max(min1, min2)
            overlap_end = min(max1, max2)
            overlap_len = overlap_end - overlap_start
            
            if overlap_len < min_depth:
                min_depth = overlap_len
                
        return min_depth

    def boundary_violation_depth(verts_inner, s_outer):
        """
        Calculate the maximum penetration depth of the inner hexagon vertices
        into the outer hexagon (centered at 0,0, side s_outer, angle 0).
        """
        max_viol = 0.0
        inradius_outer = s_outer * sqrt3 / 2
        
        # Outer hexagon normals (angle 0)
        outer_axes = []
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            outer_axes.append((math.cos(theta), math.sin(theta)))
            
        for p in verts_inner:
            px, py = p
            for ax in outer_axes:
                proj = px * ax[0] + py * ax[1]
                if proj > inradius_outer:
                    viol = proj - inradius_outer
                    if viol > max_viol:
                        max_viol = viol
        return max_viol

    def objective(params):
        # params: 2*N positions + N angles + 1 side
        pos = []
        verts_list = []
        axes_list = []
        
        for i in range(N):
            px = params[2*i]
            py = params[2*i+1]
            ang = params[2*N + i]
            pos.append((px, py))
            verts = get_hex_vertices(pos[i], ang)
            verts_list.append(verts)
            axes_list.append(get_hex_axes(pos[i], ang))
            
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                # Quick bounding box / distance check
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                if dx*dx + dy*dy > 4.0: # dist > 2
                    continue
                depth = polygon_overlap_depth(verts_list[i], axes_list[i], verts_list[j], axes_list[j])
                if depth > 0:
                    penalty += depth * 100.0 # Weight for overlap
        
        # Boundary violations
        for i in range(N):
            viol = boundary_violation_depth(verts_list[i], s)
            if viol > 0:
                penalty += viol * 1000.0 # Higher weight for boundary to ensure containment
        
        return s + penalty

    # Multi-start configurations
    best_solution = None
    best_score = float('inf')
    
    def generate_initial_config(seed):
        random.seed(seed)
        # Try a few patterns
        pattern = random.choice([0, 1, 2])
        
        pos = []
        angs = []
        
        if pattern == 0:
            # Honeycomb grid 3-4-4 or similar
            # Generate a dense cluster
            # Center at 0,0
            # Hex grid coordinates
            # For N=11, maybe 3 rows: 3, 4, 4? Or 4, 3, 4?
            # Let's try 4-3-4
            rows = [4, 3, 4]
            y_offset = 0.0
            for i, count in enumerate(rows):
                x_start = -(count - 1) * sqrt3 / 2
                for j in range(count):
                    x = x_start + j * sqrt3
                    y = y_offset
                    # Add jitter
                    x += random.uniform(-0.2, 0.2)
                    y += random.uniform(-0.2, 0.2)
                    pos.append([x, y])
                    angs.append(random.uniform(-15, 15))
                y_offset += 1.5
        elif pattern == 1:
            # 5-1-5? No, too wide.
            # 2-4-5?
            # Let's try a compact blob
            # 1 center, 6 around, 4 more?
            # Just use a grid 3x4 with one missing?
            # 4-4-3
            rows = [4, 4, 3]
            y_offset = 0.0
            for i, count in enumerate(rows):
                x_start = -(count - 1) * sqrt3 / 2
                for j in range(count):
                    x = x_start + j * sqrt3
                    y = y_offset
                    x += random.uniform(-0.2, 0.2)
                    y += random.uniform(-0.2, 0.2)
                    pos.append([x, y])
                    angs.append(random.uniform(-15, 15))
                y_offset += 1.5
        else:
            # Random scatter in a circle
            for i in range(N):
                r = random.uniform(0, 2.0)
                theta = random.uniform(0, 2*math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                pos.append([x, y])
                angs.append(random.uniform(-90, 90))
                
        # Center the configuration
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pos:
            p[0] -= cx
            p[1] -= cy
            
        return pos, angs

    # Run multiple DE starts
    num_starts = 5
    for start_idx in range(num_starts):
        pos_init, ang_init = generate_initial_config(42 + start_idx)
        
        bounds = []
        for i in range(N):
            bounds.append((-4, 4))
            bounds.append((-4, 4))
        for i in range(N):
            bounds.append((-90, 90))
        bounds.append((3.0, 6.0)) # Side length bound
        
        x0 = []
        for i in range(N):
            x0.append(pos_init[i][0])
            x0.append(pos_init[i][1])
        for i in range(N):
            x0.append(ang_init[i])
        
        # Estimate initial S
        max_dist = 0
        for p in pos_init:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        # Add margin for the hexagon radius
        x0.append(max_dist + 1.5)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=200,
                popsize=30,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42 + start_idx
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
        except Exception as e:
            pass

    if best_solution is None:
        # Fallback
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

    # Extract solution
    pos = []
    for i in range(N):
        pos.append([best_solution[2*i], best_solution[2*i+1]])
    angs = [best_solution[2*N + i] for i in range(N)]
    s = best_solution[3*N]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END
