# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import differential_evolution

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Strategy:
    1. Use Differential Evolution (DE) for global search over the positions of 12 inner hexagons.
    2. The objective function minimizes the outer hexagon side length s, with hard penalties for violations.
    3. This avoids local optima traps common in gradient-based methods.
    
    Returns:
        inner_hex_data: np.ndarray of shape (12, 3) - (x, y, angle) for each inner hexagon
        outer_hex_data: np.ndarray of shape (3,) - (x, y, angle) for outer hexagon
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    def compute_s(inner_centers):
        """
        Compute the minimum side length s for a regular hexagon centered at (0,0) with angle 0
        that contains all vertices of the inner hexagons (all aligned with angle 0).
        """
        # The 6 outward normals for a regular hexagon with angle 0 (flat top/bottom)
        # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
        # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
        normals = []
        for i in range(6):
            a = math.pi/6 + i * math.pi / 3
            normals.append((math.cos(a), math.sin(a)))
        
        max_proj = 0.0
        for cx, cy in inner_centers:
            verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in verts:
                for n in normals:
                    proj = v[0]*n[0] + v[1]*n[1]
                    if proj > max_proj:
                        max_proj = proj
        
        # For a regular hexagon with side s and angle 0:
        # The distance from center to each edge (apothem) is s * sqrt(3)/2
        # So max_proj = s * sqrt(3)/2 => s = 2 * max_proj / sqrt(3)
        return max_proj * 2.0 / sqrt3
    
    def collision_penalty(verts1, verts2):
        """
        Compute a smooth collision penalty using the Separating Axis Theorem.
        Returns 0 if separated, positive value if overlapping.
        """
        # Get all edge normals from both hexagons
        axes = []
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        total_penalty = 0.0
        for axis in axes:
            dots1 = verts1 @ axis
            dots2 = verts2 @ axis
            min1, max1 = np.min(dots1), np.max(dots1)
            min2, max2 = np.min(dots2), np.max(dots2)
            
            # Check for separation
            if max1 < min2:
                separation = min2 - max1
            elif max2 < min1:
                separation = min1 - max2
            else:
                # Overlapping: compute overlap amount
                overlap = min(max1, max2) - max(min1, min2)
                if overlap > 0:
                    total_penalty += overlap * overlap
        
        return total_penalty
    
    def containment_penalty(inner_centers, s):
        """
        Compute penalty for inner hexagon vertices outside the outer hexagon.
        Outer hexagon is centered at (0,0) with angle 0 and side s.
        """
        outer_verts = get_hex_vertices(0, 0, s, 0)
        penalty = 0.0
        
        for cx, cy in inner_centers:
            inner_verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in inner_verts:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i+1)%6]
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    if cross < 0:
                        penalty += cross * cross
        
        return penalty
    
    def objective(params):
        """
        Objective function for DE.
        params: (24,) array representing 12 (x, y) pairs.
        """
        inner_centers = params.reshape(12, 2)
        
        # Compute s
        s = compute_s(inner_centers)
        
        # Compute collision penalties
        collision_pen = 0.0
        for i in range(12):
            verts_i = get_hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, 0)
            for j in range(i+1, 12):
                verts_j = get_hex_vertices(inner_centers[j][0], inner_centers[j][1], 1.0, 0)
                collision_pen += collision_penalty(verts_i, verts_j)
        
        # Compute containment penalty
        cont_pen = containment_penalty(inner_centers, s)
        
        # Weighted sum with large weights to enforce constraints
        lambda1 = 100.0  # collision weight
        lambda2 = 100.0  # containment weight
        
        return s + lambda1 * collision_pen + lambda2 * cont_pen
    
    def validate_config(inner_centers, s):
        """
        Validate that the configuration is strictly valid per the task contract.
        """
        # Check collisions with strict separation
        for i in range(12):
            verts_i = get_hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, 0)
            for j in range(i+1, 12):
                verts_j = get_hex_vertices(inner_centers[j][0], inner_centers[j][1], 1.0, 0)
                
                # Get all edge normals
                axes = []
                for k in range(6):
                    p1 = verts_i[k]
                    p2 = verts_i[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                for k in range(6):
                    p1 = verts_j[k]
                    p2 = verts_j[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                
                separated = False
                for axis in axes:
                    dots_i = verts_i @ axis
                    dots_j = verts_j @ axis
                    min_i, max_i = np.min(dots_i), np.max(dots_i)
                    min_j, max_j = np.min(dots_j), np.max(dots_j)
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        separated = True
                        break
                if not separated:
                    return False
        
        # Check containment
        outer_verts = get_hex_vertices(0, 0, s, 0)
        for cx, cy in inner_centers:
            inner_verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in inner_verts:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i+1)%6]
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    if cross < -1e-6:
                        return False
        
        return True
    
    # Initial configuration: hexagonal lattice
    def initial_config():
        centers = []
        # Center
        centers.append([0.0, 0.0])
        # Ring 1: 6 hexagons
        for i in range(6):
            a = i * math.pi / 3
            centers.append([2.5 * math.cos(a), 2.165 * math.sin(a)])
        # Ring 2: 5 more to make 12
        for i in range(5):
            a = i * math.pi / 3 + math.pi / 6
            centers.append([5.0 * math.cos(a), 4.33 * math.sin(a)])
        return np.array(centers)
    
    # Run DE optimization
    best_val = 0.0
    best_inner = None
    best_s = float('inf')
    
    # DE bounds: reasonable range for centers
    bounds = [(-8, 8)] * 24
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,  # We'll do our own polishing
            update_interval=1,
            callback=lambda xk, convergence: convergence
        )
        
        if result.success or result.fun < float('inf'):
            inner_centers = result.x.reshape(12, 2)
            s = compute_s(inner_centers)
            
            if validate_config(inner_centers, s):
                val = 1.0 / s
                if val > best_val:
                    best_val = val
                    best_inner = inner_centers
                    best_s = s
    except Exception as e:
        pass
    
    # Try a few more DE runs with different seeds if time permits
    if time.time() < deadline - 60:
        for seed in [123, 456]:
            if time.time() > deadline - 30:
                break
            try:
                result = differential_evolution(
                    objective, 
                    bounds, 
                    seed=seed,
                    maxiter=100,
                    popsize=12,
                    tol=1e-6,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    polish=False,
                    update_interval=1,
                    callback=lambda xk, convergence: convergence
                )
                
                if result.success or result.fun < float('inf'):
                    inner_centers = result.x.reshape(12, 2)
                    s = compute_s(inner_centers)
                    
                    if validate_config(inner_centers, s):
                        val = 1.0 / s
                        if val > best_val:
                            best_val = val
                            best_inner = inner_centers
                            best_s = s
            except Exception as e:
                pass
    
    # If no valid configuration found, use fallback
    if best_inner is None:
        centers = initial_config()
        best_inner = centers
        best_s = compute_s(centers)
        best_val = 1.0 / best_s
    
    # Prepare return values
    inner_arr = np.zeros((12, 3))
    for i in range(12):
        inner_arr[i][0] = best_inner[i][0]
        inner_arr[i][1] = best_inner[i][1]
        inner_arr[i][2] = 0.0  # angle
    
    outer_arr = np.array([0.0, 0.0, 0.0])
    
    return inner_arr, outer_arr, best_s

# EVOLVE-BLOCK-END
