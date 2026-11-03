"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_covering_radius(points):
    """Compute the covering radius (in degrees) for points on unit sphere."""
    try:
        hull = ConvexHull(points)
        max_cos = 1.0  # cos of min angular distance = max cos
        for simplex in hull.simplices:
            verts = points[simplex]
            if len(verts) == 3:
                v1, v2, v3 = verts
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                dist = np.abs(np.dot(normal, v1))
                dist = np.clip(dist, -1.0, 1.0)
                cos_angle = dist
            else:
                v1, v2, v3 = verts[0], verts[1], verts[2]
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                dist = np.abs(np.dot(normal, v1))
                dist = np.clip(dist, -1.0, 1.0)
                cos_angle = dist
            
            max_cos = min(max_cos, cos_angle)
        
        if max_cos > 1.0:
            max_cos = 1.0
        if max_cos < -1.0:
            max_cos = -1.0
        
        radius_rad = np.arccos(max_cos)
        return np.degrees(radius_rad)
    except Exception:
        return 180.0


def _compute_sinkhorn_cost(points, n_samples=2000, epsilon=0.1, n_iter=50):
    """
    Compute a Sinkhorn-like cost between empirical measure of points and uniform measure.
    This provides a smooth surrogate for covering radius.
    """
    try:
        # Generate uniform samples on sphere
        u = np.random.uniform(0, 1, n_samples)
        v = np.random.uniform(0, 1, n_samples)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        samples = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Compute cost matrix: squared angular distance
        dots = np.clip(points @ samples.T, -1.0, 1.0)
        angles = np.arccos(dots)
        cost = angles ** 2
        
        # Sinkhorn iterations
        a = np.ones(N) / N
        b = np.ones(n_samples) / n_samples
        
        u_vec = np.zeros(N)
        v_vec = np.zeros(n_samples)
        
        for _ in range(n_iter):
            # Update u
            exp_neg_cost_over_eps = np.exp(-cost / epsilon)
            u_vec = np.log(a / (exp_neg_cost_over_eps @ v_vec + 1e-12))
            # Update v
            v_vec = np.log(b / (exp_neg_cost_over_eps.T @ np.exp(u_vec) + 1e-12))
        
        # Compute transport cost
        transport = np.exp(u_vec[:, None] + v_vec[None, :] - cost / epsilon)
        transport = transport * a[:, None]
        cost_val = np.sum(transport * cost)
        
        return cost_val
    except Exception:
        return 1e6


def _local_optimize_sinkhorn(points, deadline, n_samples=1000, epsilon=0.5, n_iter=30):
    """
    Local optimization using Sinkhorn cost as a smooth surrogate.
    Uses gradient-free approach: move each point to reduce local Sinkhorn cost.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_cost = _compute_sinkhorn_cost(pts, n_samples, epsilon, n_iter)
    
    lr = 0.005
    for iteration in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Compute pairwise distances
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # For each point, compute a force based on local density
        new_pts = pts.copy()
        for k in range(N):
            # Find nearest neighbors
            neighbor_dists = dists[k].copy()
            neighbor_dists[k] = np.inf
            nn_idx = np.argsort(neighbor_dists)[:8]
            nn_pts = pts[nn_idx]
            
            # Centroid of nearest neighbors
            centroid = nn_pts.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                continue
            centroid = centroid / centroid_norm
            
            # Move away from centroid to spread out
            # But also consider the global distribution via Sinkhorn
            # Simple heuristic: move slightly away from local centroid
            move_dir = pts[k] - centroid
            move_norm = np.linalg.norm(move_dir)
            if move_norm < 1e-12:
                move_dir = np.random.randn(3)
                move_norm = np.linalg.norm(move_dir)
            move_dir = move_dir / move_norm
            
            # Small step
            new_pts[k] = pts[k] + lr * move_dir
            new_pts[k] = new_pts[k] / np.linalg.norm(new_pts[k])
        
        # Check if this improves
        current_cost = _compute_sinkhorn_cost(new_pts, n_samples, epsilon, n_iter)
        if current_cost < best_cost:
            best_cost = current_cost
            best_pts = new_pts.copy()
            pts = new_pts.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Phase 1: Sinkhorn-based optimization (smooth surrogate)
    sinkhorn_deadline = deadline - 40.0
    if time.time() < sinkhorn_deadline:
        points = _local_optimize_sinkhorn(points, sinkhorn_deadline)
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Repulsive force optimization (refinement)
    lr = 0.01
    for iteration in range(300):
        if time.time() > deadline - 5.0:
            break
        
        new_points = points.copy()
        for idx in range(N):
            forces = np.zeros(3)
            for jdx in range(N):
                if jdx == idx:
                    continue
                diff = points[idx] - points[jdx]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    forces += np.random.randn(3) * 0.1
                else:
                    force_mag = 1.0 / (dist * dist + 0.01)
                    forces += force_mag * diff / dist
            
            force_norm = np.linalg.norm(forces)
            if force_norm > 1e-10:
                new_points[idx] = points[idx] + lr * forces / force_norm
                new_points[idx] /= np.linalg.norm(new_points[idx])
        
        points = new_points
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
