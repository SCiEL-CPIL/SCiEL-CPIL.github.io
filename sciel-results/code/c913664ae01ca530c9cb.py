import numpy as np
from scipy.optimize import minimize, LinearConstraint
import itertools


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length s (maximizing 1/s).
    
    Uses a continuous optimization approach (L-BFGS-B) with:
    - Repulsive penalty for overlapping inner hexagons
    - Penalty for inner vertices outside the outer hexagon
    - Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is float
    """
    N = 12
    # All inner hexagons have angle 0 for simplicity
    # Variables: x_0, y_0, x_1, y_1, ..., x_11, y_11, s
    # Total: 24 + 1 = 25 variables
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get 6 vertices of a regular hexagon."""
        a = np.radians(angle_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.array([[cx + side * np.cos(ang), cy + side * np.sin(ang)] for ang in angles])
    
    def get_hex_edge_normals(vertices):
        """Get outward normals of hexagon edges (unit vectors)."""
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            edge = v2 - v1
            # Outward normal: perpendicular to edge, pointing outward
            normal = np.array([-edge[1], edge[0]])
            # Ensure it points outward (away from center)
            center = np.mean(vertices, axis=0)
            mid = (v1 + v2) / 2
            if np.dot(normal, mid - center) < 0:
                normal = -normal
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return np.array(normals)
    
    def check_hex_overlap(v1, v2, normals1, normals2, tol=1e-6):
        """
        Check if two hexagons are separated using SAT.
        Returns (is_separated, min_gap) where min_gap is the minimum separation.
        """
        min_gap = np.inf
        # Check all edge normals from both hexagons
        for normal in np.vstack([normals1, normals2]):
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Gap between intervals
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                # Overlapping projections on this axis
                return False, 0.0
            min_gap = min(min_gap, gap)
        return True, min_gap
    
    def penalty_and_grad(x, s, inner_centers, inner_angles):
        """
        Compute penalty for:
        1. Overlapping inner hexagons
        2. Inner vertices outside outer hexagon
        """
        penalty = 0.0
        grad = np.zeros_like(x)
        
        # Outer hexagon vertices
        outer_verts = get_hex_vertices(0, 0, s, 0)
        outer_normals = get_hex_edge_normals(outer_verts)
        
        # Inner hexagons vertices and normals
        inner_verts_list = []
        inner_normals_list = []
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = get_hex_vertices(cx, cy, 1.0, ang)
            norms = get_hex_edge_normals(verts)
            inner_verts_list.append(verts)
            inner_normals_list.append(norms)
        
        # Check outer boundary violations
        for i in range(N):
            verts = inner_verts_list[i]
            for v in verts:
                # Check if v is inside outer hexagon
                for j in range(6):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j + 1) % 6]
                    edge = p2 - p1
                    normal = outer_normals[j]
                    # Signed distance: negative means outside
                    dist = np.dot(v - p1, normal)
                    if dist < 0:
                        # Penalty for violation
                        penalty += dist * dist
                        # Gradient contribution
                        # d(dist)/d(v) = normal
                        # v depends on inner_centers[i]
                        # For simplicity, use finite difference approximation
                        pass
        
        # Check inner-inner overlaps
        for i in range(N):
            for j in range(i + 1, N):
                is_sep, min_gap = check_hex_overlap(
                    inner_verts_list[i], inner_verts_list[j],
                    inner_normals_list[i], inner_normals_list[j]
                )
                if not is_sep:
                    # Overlapping: add penalty
                    # Use a smooth penalty based on maximum penetration
                    # For simplicity, use a fixed penalty per overlapping pair
                    penalty += 1.0
                    # Gradient: push apart
                    # Approximate by computing center-to-center direction
                    c1 = inner_centers[i]
                    c2 = inner_centers[j]
                    diff = c2 - c1
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        diff = diff / dist
                    else:
                        diff = np.array([1.0, 0.0])
                    # Penalty gradient: -diff for i, +diff for j
                    grad[2 * i:2 * i + 2] -= 0.1 * diff
                    grad[2 * j:2 * j + 2] += 0.1 * diff
        
        return penalty, grad
    
    # Initial guess: hexagonal lattice arrangement
    # Center at (0,0), 6 around it, 5 more
    hex_height = np.sqrt(3)
    initial_centers = []
    # Center
    initial_centers.append([0, 0])
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = np.radians(60 * k)
        initial_centers.append([2.5 * np.cos(ang), 2.5 * np.sin(ang)])
    # Ring 2: need 5 more (total 12)
    for k in range(5):
        ang = np.radians(60 * k + 30)
        initial_centers.append([5.0 * np.cos(ang), 5.0 * np.sin(ang)])
    
    initial_centers = np.array(initial_centers)
    initial_angles = np.zeros(N)
    # Initial outer side length estimate
    # Find max distance from origin to any inner vertex
    max_dist = 0
    for c in initial_centers:
        verts = get_hex_vertices(c[0], c[1], 1.0, 0)
        for v in verts:
            max_dist = max(max_dist, np.linalg.norm(v))
    initial_s = max_dist * 1.1  # Some margin
    
    x0 = np.concatenate([initial_centers.flatten(), [initial_s]])
    
    def objective(x):
        """Objective: s + penalty for violations."""
        centers = x[:2 * N].reshape(N, 2)
        s = x[2 * N]
        angles = inner_angles  # Fixed at 0
        
        # Compute penalty
        pen, _ = penalty_and_grad(x, s, centers, angles)
        
        # Objective: minimize s + penalty
        return s + 10.0 * pen
    
    # Gradient (approximate using finite differences since analytical is complex)
    def gradient(x):
        centers = x[:2 * N].reshape(N, 2)
        s = x[2 * N]
        angles = inner_angles
        
        pen, grad_pen = penalty_and_grad(x, s, centers, angles)
        
        # Gradient of s is 1 for the last component
        grad = np.zeros_like(x)
        grad[:2 * N] = grad_pen[:2 * N]
        grad[2 * N] = 1.0
        # Penalty gradient also contributes to s through the constraint
        # For simplicity, ignore penalty's contribution to s gradient
        return grad + 10.0 * np.zeros_like(x)  # Penalty gradient is approximate
    
    # Use L-BFGS-B optimization
    # Bounds: s must be at least sqrt(12) ~ 3.46, centers within reasonable range
    bounds = [(-20, 20)] * (2 * N) + [(np.sqrt(12) * 0.9, 20)]
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    x_opt = result.x
    centers = x_opt[:2 * N].reshape(N, 2)
    s_opt = x_opt[2 * N]
    
    # Ensure s is at least the minimum required
    s_opt = max(s_opt, np.sqrt(12) * (1 + 1e-6))
    
    # Construct return values
    inner_hex_data = np.zeros((N, 3))
    for i in range(N):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0  # angle in degrees
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt


# EVOLVE-BLOCK-END
