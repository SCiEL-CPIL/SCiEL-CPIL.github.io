"""Rewritten program: hexagonal lattice init + force-directed local optimization + spectral polish."""
import numpy as np
import time
from scipy.spatial import Delaunay

N = 300
DEADLINE = 110.0  # seconds, safety margin before 120s limit


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return N closest to center."""
    # Optimal hexagonal packing spacing for 300 circles
    # We want to find a good initial spacing
    # For hexagonal packing, area per circle = sqrt(3)/2 * d^2 where d is spacing
    # Total area = pi, so 300 * sqrt(3)/2 * d^2 ≈ pi * (1-r)^2
    # r ≈ 0.053, so (1-r)^2 ≈ 0.893, pi*0.893 ≈ 2.805
    # 300 * 0.866 * d^2 ≈ 2.805 => d^2 ≈ 0.01076 => d ≈ 0.1037
    
    # Start with a reasonable spacing and refine
    best_points = None
    for spacing in [0.10, 0.102, 0.104, 0.106, 0.108, 0.11, 0.112]:
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            # Sort by distance from center, take N closest
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
        else:
            if best_points is None:
                best_points = points
    
    if best_points is None:
        # Fallback: uniform grid
        best_points = np.random.randn(N, 2) * 0.5
    
    return best_points[:N]


def _compute_radius(points):
    """Compute the minimum radius: min of half min distance and min distance to boundary."""
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    
    return min(min_dist / 2.0, min_boundary)


def _force_directed_optimize(points, max_iters=5000, dt=0.001):
    """Force-directed optimization to maximize minimum radius."""
    points = points.copy()
    n = len(points)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Repulsion between all pairs (only need to consider close pairs for efficiency)
        # For N=300, O(N^2) = 90000 pairs, which is fine
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Force: push apart if too close
        # Use a soft force that scales with proximity
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d < 0.1:  # Only consider close pairs for efficiency
                    # Force proportional to 1/d^2 (Coulomb-like)
                    f = 1.0 / (d * d + 1e-10)
                    direction = diff[i, j] / (d + 1e-10)
                    forces[i] += direction * f
                    forces[j] -= direction * f
        
        # Boundary force: push inward if near boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        for i in range(n):
            if radii[i] > 0.95:
                # Push toward center
                f = (radii[i] - 0.95) * 5.0
                direction = -points[i] / (radii[i] + 1e-10)
                forces[i] += direction * f
        
        # Update positions with adaptive step size
        # Normalize forces
        force_mags = np.sqrt(np.sum(forces ** 2, axis=1))
        max_force = np.max(force_mags)
        if max_force > 0:
            # Adaptive step size
            current_r = _compute_radius(points)
            step = dt / (1.0 + max_force * 0.1)
            points += forces * step
        
        # Ensure points stay inside the unit circle
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / radii[mask]
    
    return points


def _spectral_polish(points, max_iters=1000):
    """Use Delaunay triangulation and graph Laplacian to enforce uniformity."""
    points = points.copy()
    n = len(points)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
        
        try:
            # Compute Delaunay triangulation
            tri = Delaunay(points)
            
            # Build adjacency matrix from Delaunay
            adj = np.zeros((n, n))
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        adj[simplex[i], simplex[j]] = 1
                        adj[simplex[j], simplex[i]] = 1
            
            # Compute graph Laplacian
            degree = adj.sum(axis=1)
            L = np.diag(degree) - adj
            
            # Compute eigenvalues of L (algebraic connectivity is second smallest)
            # For efficiency, use power iteration or scipy
            from scipy.linalg import eigh
            # Only compute a few eigenvalues
            try:
                vals, vecs = eigh(L, subset_by_index=[1, min(5, n-1)])
            except:
                vals, vecs = eigh(L)
                vals = vals[1:6]
                vecs = vecs[:, 1:6]
            
            # Use the Fiedler vector (second eigenvector) to identify poorly connected regions
            # and apply corrective forces
            if len(vecs) > 0:
                fiedler = vecs[:, 0]
                # Points with large |fiedler| value are at the "edges" of the structure
                # Apply gentle forces to smooth out variations
                target = np.mean(fiedler ** 2)
                forces = np.zeros_like(points)
                for i in range(n):
                    # Force toward reducing variance in fiedler values
                    f = (fiedler[i] ** 2 - target) * 0.01
                    # Direction: toward center of mass of nearby points
                    neighbors = np.where(adj[i] > 0)[0]
                    if len(neighbors) > 0:
                        center = points[neighbors].mean(axis=0)
                        direction = center - points[i]
                        forces[i] += direction * f / (np.linalg.norm(direction) + 1e-10)
                
                if np.any(forces != 0):
                    points += forces * 0.01
        except Exception:
            break
        
        # Keep points inside unit circle
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / radii[mask]
    
    return points


def _final_polish(points, max_iters=2000):
    """High-precision final polish using gradient descent on the objective."""
    points = points.copy()
    n = len(points)
    
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    lr = 0.0001
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 1:
            break
        
        # Compute gradient of -r (we want to maximize r)
        # r = min(min_pair_dist/2, min_boundary_dist)
        # This is non-smooth, so use a soft approximation
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Soft min using log-sum-exp
        # For pairwise distances
        min_pair_dist = np.min(dists)
        min_boundary = np.min(boundary_dists)
        
        # Find which constraint is active
        if min_pair_dist / 2.0 < min_boundary:
            # Pairwise constraint is active
            # Find the minimum pair
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx
            if i > j:
                i, j = j, i
            
            # Gradient: push i and j apart
            grad = np.zeros_like(points)
            direction = points[i] - points[j]
            d = np.linalg.norm(direction)
            if d > 1e-10:
                direction /= d
                grad[i] = -direction * 0.5
                grad[j] = direction * 0.5
            
            # Also add small repulsion from all close pairs
            for a in range(n):
                for b in range(a + 1, n):
                    d_ab = dists[a, b]
                    if d_ab < min_pair_dist * 1.5:
                        dir_ab = points[a] - points[b]
                        norm_ab = np.linalg.norm(dir_ab)
                        if norm_ab > 1e-10:
                            dir_ab /= norm_ab
                            weight = 1.0 / (d_ab + 1e-10)
                            grad[a] += dir_ab * weight * 0.1
                            grad[b] -= dir_ab * weight * 0.1
        else:
            # Boundary constraint is active
            min_boundary_idx = np.argmin(boundary_dists)
            i = min_boundary_idx
            
            grad = np.zeros_like(points)
            # Push inward
            r_i = radii[i]
            if r_i > 1e-10:
                direction = -points[i] / r_i
                grad[i] = direction
            
            # Also add small repulsion from all close pairs
            for a in range(n):
                for b in range(a + 1, n):
                    d_ab = dists[a, b]
                    if d_ab < min_pair_dist * 1.5:
                        dir_ab = points[a] - points[b]
                        norm_ab = np.linalg.norm(dir_ab)
                        if norm_ab > 1e-10:
                            dir_ab /= norm_ab
                            weight = 1.0 / (d_ab + 1e-10)
                            grad[a] += dir_ab * weight * 0.1
                            grad[b] -= dir_ab * weight * 0.1
        
        # Update
        points += lr * grad
        
        # Keep inside unit circle
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / radii[mask]
        
        # Check improvement
        current_r = _compute_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    return best_points


def construct():
    """Construct 300 circle centers in the unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Step 1: Initialize with hexagonal lattice
    points = _hex_lattice_init()
    
    # Step 2: Force-directed optimization
    points = _force_directed_optimize(points, max_iters=3000, dt=0.002)
    
    # Step 3: Spectral polish
    points = _spectral_polish(points, max_iters=500)
    
    # Step 4: Final high-precision polish
    points = _final_polish(points, max_iters=1000)
    
    # Ensure validity: all points inside unit circle
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii > 0.9999
    if np.any(mask):
        points[mask] *= 0.9999 / radii[mask]
    
    # Ensure exactly N points
    if len(points) != N:
        if len(points) < N:
            # Add random points inside the disk
            while len(points) < N:
                p = np.random.randn(2) * 0.5
                if np.hypot(p[0], p[1]) < 0.9:
                    points = np.vstack([points, p])
        else:
            points = points[:N]
    
    return points[:N]
