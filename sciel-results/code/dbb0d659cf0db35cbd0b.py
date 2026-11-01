import numpy as np
from scipy.optimize import differential_evolution
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Helper functions for geometry
    def hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        verts = np.array([
            [cx + side * math.cos(angle + i * math.pi / 3),
             cy + side * math.sin(angle + i * math.pi / 3)]
            for i in range(6)
        ])
        return verts

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
        # Pre-compute edges and normals
        # v1 and v2 are (6,2) arrays
        edges1 = np.roll(v1, -1, axis=0) - v1
        edges2 = np.roll(v2, -1, axis=0) - v2
        
        # Normals (perpendicular to edges)
        norms1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        norms2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        
        # Normalize normals
        norms1 = norms1 / (np.linalg.norm(norms1, axis=1, keepdims=True) + 1e-9)
        norms2 = norms2 / (np.linalg.norm(norms2, axis=1, keepdims=True) + 1e-9)
        
        # Project vertices onto normals
        # Check all 12 axes
        for n in np.vstack([norms1, norms2]):
            proj1 = v1 @ n
            proj2 = v2 @ n
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they are disjoint
            if max1 < min2 or max2 < min1:
                return False
        
        return True

    def is_inside(v_inner, v_outer):
        """Check if inner polygon is inside outer polygon using SAT."""
        edges_in = np.roll(v_inner, -1, axis=0) - v_inner
        edges_out = np.roll(v_outer, -1, axis=0) - v_outer
        
        norms_in = np.column_stack([-edges_in[:, 1], edges_in[:, 0]])
        norms_out = np.column_stack([-edges_out[:, 1], edges_out[:, 0]])
        
        norms_in = norms_in / (np.linalg.norm(norms_in, axis=1, keepdims=True) + 1e-9)
        norms_out = norms_out / (np.linalg.norm(norms_out, axis=1, keepdims=True) + 1e-9)
        
        for n in np.vstack([norms_in, norms_out]):
            proj_in = v_inner @ n
            proj_out = v_outer @ n
            
            min_in, max_in = np.min(proj_in), np.max(proj_in)
            min_out, max_out = np.min(proj_out), np.max(proj_out)
            
            # Inner must be within outer's projection
            if max_in > max_out or min_in < min_out:
                return False
        
        return True

    def evaluate(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, side]
        """
        if time.time() > deadline:
            return 1e6
            
        inner_params = params[:3*N]
        outer_side = params[3*N]
        
        # Outer hexagon is fixed at center (0,0) and angle 0
        v_outer = hex_vertices(0, 0, outer_side, 0)
        
        inner_verts_list = []
        for i in range(N):
            x = inner_params[3*i]
            y = inner_params[3*i+1]
            a = inner_params[3*i+2]
            v = hex_vertices(x, y, 1.0, a)
            inner_verts_list.append(v)
            
        # Check containment
        for v in inner_verts_list:
            if not is_inside(v, v_outer):
                return 1e6 + outer_side # Penalize and return side to guide search? 
                # Actually, just return large penalty.
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if sat_overlap(inner_verts_list[i], inner_verts_list[j]):
                    return 1e6 + outer_side
        
        return outer_side

    # Define bounds
    # x, y in [-5, 5], angle in [0, 60], side in [3.0, 5.0]
    bounds = [(-5, 5), (-5, 5), (0, 60)] * N + [(3.0, 5.0)]
    
    # Initial guess: Honeycomb pattern
    # 1 center, 6 ring 1, 5 from ring 2
    init = []
    init += [0.0, 0.0, 0.0]
    for k in range(6):
        angle = k * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        init += [x, y, 0.0]
    # 5 more, placed roughly in the second ring
    # Ring 2 distance is approx 2*sqrt(3)?
    # Let's place them at angles offset by 30 deg, distance 2*sqrt(3)
    for k in range(5):
        angle = k * 2 * math.pi / 5 + math.pi/10
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        init += [x, y, 0.0]
    init += [4.0] # Initial side guess
    
    # Differential Evolution
    # To save time, we use a small population and few generations
    # We can also try a local refinement afterwards
    try:
        res = differential_evolution(
            evaluate, 
            bounds, 
            x0=init,
            seed=42,
            maxiter=50, 
            popsize=10,
            tol=1e-4,
            polish=True # Use L-BFGS-B for final polish
        )
        best_side = res.fun
        best_params = res.x
    except Exception as e:
        # Fallback to initial guess if DE fails
        best_side = evaluate(init)
        best_params = init

    # Final check and extraction
    if best_side > 1e5:
        # Invalid, return a large valid box
        inner = []
        for i in range(N):
            inner.append([float(i*2.0), 0.0, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(x), float(y), float(a)])
    
    center = (0.0, 0.0)
    side = float(best_params[3*N])
    angle = 0.0
    
    return inner, center, side, angle

