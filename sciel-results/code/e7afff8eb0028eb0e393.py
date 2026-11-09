import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """Construct packing of N unit hexagons inside a regular hexagon."""
    
    def hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3.0
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    def min_outer_side(inner_hexes, outer_center, outer_rotation_deg):
        ox, oy = outer_center
        outer_rot = math.radians(outer_rotation_deg)
        apothem_factor = math.sqrt(3.0) / 2.0
        max_proj = 0.0
        for (cx, cy, ang_deg) in inner_hexes:
            verts = hex_vertices(cx, cy, 1.0, ang_deg)
            for (px, py) in verts:
                for k in range(6):
                    theta = outer_rot + k * math.pi / 3.0
                    proj = (px - ox) * math.cos(theta) + (py - oy) * math.sin(theta)
                    if proj > max_proj:
                        max_proj = proj
        return max_proj / apothem_factor
    
    # Initial arrangement: hexagonal lattice
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = k * math.pi / 3.0
        x = 2.0 * math.cos(a)
        y = 2.0 * math.sin(a)
        inner.append((x, y, 0.0))
    for k in range(6):
        a = k * math.pi / 3.0 + math.pi / 6.0
        x = 4.0 * math.cos(a)
        y = 4.0 * math.sin(a)
        inner.append((x, y, 0.0))
    inner = inner[:N]
    
    outer_center = (0.0, 0.0)
    outer_rot = 0.0
    
    np.random.seed(42)
    
    best_inner = [list(h) for h in inner]
    best_oc = list(outer_center)
    best_or = outer_rot
    best_score = min_outer_side(best_inner, best_oc, best_or)
    
    # Iterative local search with decreasing step size
    for iteration in range(200):
        step = 0.5 * (0.95 ** iteration)
        improved = False
        for _ in range(80):
            i = np.random.randint(N)
            new_inner = [list(h) for h in best_inner]
            dx = np.random.normal(0, step)
            dy = np.random.normal(0, step)
            da = np.random.normal(0, step * 30)
            new_inner[i][0] += dx
            new_inner[i][1] += dy
            new_inner[i][2] += da
            
            new_oc = [best_oc[0] + np.random.normal(0, step * 0.5), 
                      best_oc[1] + np.random.normal(0, step * 0.5)]
            new_or = best_or + np.random.normal(0, step * 10)
            
            score = min_outer_side(new_inner, new_oc, new_or)
            if score < best_score:
                best_inner = new_inner
                best_oc = new_oc
                best_or = new_or
                best_score = score
                improved = True
        
        if not improved:
            break
    
    inner_result = [[h[0], h[1], h[2]] for h in best_inner]
    return inner_result, (best_oc[0], best_oc[1]), best_score, best_or
# EVOLVE-BLOCK-END
