"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the packing radius for a set of centers in [-0.5, 0.5]^3."""
    if len(centers) < 2:
        # Handle edge case
        min_dist = np.min([0.5 - abs(c) for c in centers[0]])
        return min_dist / 2.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    min_boundary = np.min([0.5 - abs(c) for c in centers])
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_radius(centers, beta):
    """
    Smooth surrogate for the packing radius using soft-min.
    r = soft_min(min_pair_dist/2, min_boundary_dist)
    """
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Convert to radii (half distances)
    pair_radii = np.sqrt(dists_sq) / 2.0
    
    # Boundary distances
    boundary_radii = 0.5 - np.abs(centers)
    boundary_min = np.min(boundary_radii, axis=1)
    
    # Flatten pair radii (upper triangle only to avoid duplicates, but we can use all)
    # Actually, let's just use all off-diagonal elements
    mask = ~np.eye(len(centers), dtype=bool)
    pair_flat = pair_radii[mask]
    
    # Combine all constraints
    all_radii = np.concatenate([pair_flat, boundary_min])
    
    # Soft min: -log(sum(exp(-x/beta))) * beta
    # To avoid overflow, subtract max
    max_val = np.max(all_radii)
    exp_vals = np.exp(-(all_radii - max_val) / beta)
    log_sum = np.log(np.sum(exp_vals))
    soft_min = max_val - beta * log_sum
    
    return soft_min


def _soft_min_radius_grad(centers, beta):
    """
    Gradient of the soft-min radius with respect to centers.
    """
    n = len(centers)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Convert to radii
    pair_radii = np.sqrt(dists_sq) / 2.0
    
    # Boundary distances
    boundary_radii = 0.5 - np.abs(centers)
    boundary_min = np.min(boundary_radii, axis=1)
    
    # Flatten pair radii
    mask = ~np.eye(n, dtype=bool)
    pair_flat = pair_radii[mask]
    
    # Combine all constraints
    all_radii = np.concatenate([pair_flat, boundary_min])
    
    # Soft min computation
    max_val = np.max(all_radii)
    exp_vals = np.exp(-(all_radii - max_val) / beta)
    log_sum = np.log(np.sum(exp_vals))
    soft_min = max_val - beta * log_sum
    
    # Gradient computation
    # d(soft_min)/d(all_radii) = -exp_vals / sum(exp_vals)
    weights = -exp_vals / np.sum(exp_vals)
    
    # Now we need to propagate gradients back to centers
    grad = np.zeros_like(centers)
    
    # Pairwise contribution
    # For each pair (i, j), the radius is |c_i - c_j| / 2
    # d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (4 * r_ij) where r_ij is the distance
    # Actually, d(dist/2)/d(c_i) = (c_i - c_j) / (2 * dist) * 1/2? No.
    # Let r_ij = |c_i - c_j| / 2
    # d r_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|) * 1 = (c_i - c_j) / (4 * r_ij)
    
    # We need to accumulate gradients from the flattened pair radii
    # The mask gives us the indices in row-major order of the upper/lower triangle?
    # Actually, mask is all off-diagonal, so it includes both (i,j) and (j,i).
    # This is inefficient but correct if we handle it properly.
    
    # Let's use a different approach: compute gradients directly
    # For each pair (i, j) with i < j:
    # Contribution to grad[i] += weights[k] * (c_i - c_j) / (4 * r_ij)
    # Contribution to grad[j] += weights[k] * (c_j - c_i) / (4 * r_ij)
    
    # But we need to map the flattened indices back to (i, j) pairs
    # This is complex. Let's use a simpler but potentially less efficient approach.
    
    # Alternative: use automatic differentiation or just do a finite difference check
    # For now, let's implement a direct gradient computation
    
    # Get indices of non-diagonal elements
    i_idx, j_idx = np.where(mask)
    
    # For each pair, compute the gradient contribution
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = weights[k]
        
        if w == 0:
            continue
            
        vec = centers[i] - centers[j]
        dist = np.sqrt(np.sum(vec**2))
        if dist < 1e-15:
            continue
            
        # d r_ij / d c_i = vec / (2 * dist) * 1/2? 
        # r_ij = |c_i - c_j| / 2
        # d r_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
        # Wait, derivative of |x|/2 w.r.t x is x/(2|x|)
        # So d r_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
        
        grad_i = vec / (2.0 * dist)
        grad[i] += w * grad_i
        grad[j] -= w * grad_i
    
    # Boundary contribution
    # For each center i, the boundary radius is min_k (0.5 - |c_ik|)
    # The gradient of min is only from the active constraint(s)
    # In soft-min, all constraints contribute with weights
    
    # For each center i:
    # boundary_radii[i] = min_k (0.5 - |c_ik|)
    # This is a soft-min over the 3 dimensions for each center
    # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    
    # We need to handle the soft-min over dimensions for each center
    # Let's compute this properly
    
    # For each center i, we have 3 boundary distances: 0.5 - |c_i0|, 0.5 - |c_i1|, 0.5 - |c_i2|
    # The soft-min of these gives the effective boundary radius for center i
    # Then the overall soft-min combines all pair radii and all center boundary radii
    
    # Let me restructure: 
    # For each center i, compute soft-min of its 3 boundary distances
    # Then combine with pair radii in the final soft-min
    
    # This is getting complex. Let me simplify by using a different approach.
    
    # Actually, let's just use the direct formula and accept that the gradient might be approximate
    # Or better, let's use L-BFGS-B with finite differences if analytic gradient is too complex
    
    return grad


def _fcc_lattice_init():
    """
    Generate FCC lattice points within the unit cube [-0.5, 0.5]^3.
    FCC lattice has points at:
    - (0, 0, 0) + integer translations
    - (0.5, 0.5, 0) + integer translations
    - (0.5, 0, 0.5) + integer translations
    - (0, 0.5, 0.5) + integer translations
    
    For a cube of side 1, we need to scale appropriately.
    The FCC lattice constant a relates to the nearest neighbor distance d by d = a/sqrt(2).
    We want to fit as many points as possible.
    """
    points = []
    
    # FCC basis vectors
    # In a cube of side L=1, the FCC unit cell has side a = L/n where n is the number of cells
    # Let's try different lattice spacings
    
    # The FCC lattice points in a cube [0, 1]^3 (then shift to [-0.5, 0.5]^3)
    # Standard FCC: points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in unit cell
    
    # Let's generate points with various lattice spacings
    for n in range(2, 10):
        a = 1.0 / n  # lattice spacing
        
        # Generate all FCC points in [0, 1]^3
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    # Type 1: (i*a, j*a, k*a)
                    x, y, z = i * a, j * a, k * a
                    if x <= 1.0 and y <= 1.0 and z <= 1.0:
                        points.append((x, y, z))
                    
                    # Type 2: (i*a + 0.5*a, j*a + 0.5*a, k*a)
                    x, y, z = (i + 0.5) * a, (j + 0.5) * a, k * a
                    if x <= 1.0 and y <= 1.0 and z <= 1.0:
                        points.append((x, y, z))
                    
                    # Type 3: (i*a + 0.5*a, j*a, k*a + 0.5*a)
                    x, y, z = (i + 0.5) * a, j * a, (k + 0.5) * a
                    if x <= 1.0 and y <= 1.0 and z <= 1.0:
                        points.append((x, y, z))
                    
                    # Type 4: (i*a, j*a + 0.5*a, k*a + 0.5*a)
                    x, y, z = i * a, (j + 0.5) * a, (k + 0.5) * a
                    if x <= 1.0 and y <= 1.0 and z <= 1.0:
                        points.append((x, y, z))
    
    # Convert to numpy array and shift to [-0.5, 0.5]^3
    points = np.array(points)
    points = points - 0.5
    
    # Remove duplicates
    if len(points) > 0:
        # Round to avoid floating point issues
        points_rounded = np.round(points, decimals=10)
        unique_mask = np.ones(len(points_rounded), dtype=bool)
        for i in range(len(points_rounded)):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, len(points_rounded)):
                if not unique_mask[j]:
                    continue
                if np.allclose(points_rounded[i], points_rounded[j], atol=1e-9):
                    unique_mask[j] = False
        points = points[unique_mask]
    
    # Sort by minimum distance to boundary (keep points far from boundary)
    if len(points) > N:
        min_dist_to_boundary = np.min(0.5 - np.abs(points), axis=1)
        # Sort by min_dist_to_boundary in descending order
        sorted_idx = np.argsort(-min_dist_to_boundary)
        points = points[sorted_idx[:N]]
    elif len(points) < N:
        # If we don't have enough FCC points, pad with simple cubic grid points
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        # Combine FCC and grid points, remove duplicates
        all_points = np.vstack([points, grid])
        # Remove duplicates
        all_points_rounded = np.round(all_points, decimals=10)
        unique_mask = np.ones(len(all_points_rounded), dtype=bool)
        for i in range(len(all_points_rounded)):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, len(all_points_rounded)):
                if not unique_mask[j]:
                    continue
                if np.allclose(all_points_rounded[i], all_points_rounded[j], atol=1e-9):
                    unique_mask[j] = False
        all_points = all_points[unique_mask]
        points = all_points[:N]
    
    return points


def construct():
    """
    Construct centers of 100 equal spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initial guess from FCC lattice
    centers = _fcc_lattice_init()
    centers = centers.astype(np.float64)
    
    # Ensure valid initial points
    if len(centers) != N:
        # Fallback to simple cubic grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].astype(np.float64)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Optimization using L-BFGS-B with soft-min objective
    # We'll minimize the negative soft-min radius
    
    def objective_and_grad(x):
        """Objective function and gradient for optimization."""
        elapsed = time.time() - start_time
        if elapsed > deadline:
            raise Exception("Deadline exceeded")
        
        centers_reshaped = x.reshape(N, 3)
        
        # Adaptive beta: start large, decrease over time
        progress = min(elapsed / 100.0, 1.0)
        beta = 0.05 * (1.0 - progress) + 0.001 * progress
        
        # Compute soft-min radius
        r = _soft_min_radius(centers_reshaped, beta)
        
        # Compute gradient
        grad_flat = np.zeros_like(x)
        
        # Pairwise gradient
        diff = centers_reshaped[:, np.newaxis, :] - centers_reshaped[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        pair_radii = dists / 2.0
        
        # Boundary distances
        boundary_radii = 0.5 - np.abs(centers_reshaped)
        boundary_min = np.min(boundary_radii, axis=1)
        
        # Flatten pair radii
        mask = ~np.eye(N, dtype=bool)
        pair_flat = pair_radii[mask]
        
        # Combine all constraints
        all_radii = np.concatenate([pair_flat, boundary_min])
        
        # Soft min weights
        max_val = np.max(all_radii)
        exp_vals = np.exp(-(all_radii - max_val) / beta)
        sum_exp = np.sum(exp_vals)
        weights = -exp_vals / sum_exp
        
        # Pairwise gradient
        i_idx, j_idx = np.where(mask)
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            w = weights[k]
            if abs(w) < 1e-15:
                continue
            vec = centers_reshaped[i] - centers_reshaped[j]
            dist = dists[i, j]
            if dist < 1e-15:
                continue
            grad_vec = vec / (2.0 * dist)
            grad_flat[i * 3:(i + 1) * 3] += w * grad_vec
            grad_flat[j * 3:(j + 1) * 3] -= w * grad_vec
        
        # Boundary gradient
        # For each center, the boundary constraint is min over 3 dimensions
        # We need to compute the soft-min gradient for each center's 3 boundary distances
        for i in range(N):
            # Get the weights for this center's boundary constraints
            # The boundary_min for center i is at index len(pair_flat) + i in all_radii
            # But we need the soft-min over the 3 dimensions, not just the min
            # This is tricky. Let's approximate by using the active constraint
            # Find which dimension gives the minimum boundary distance
            min_dim = np.argmin(boundary_radii[i])
            w_boundary = weights[len(pair_flat) + i]
            if abs(w_boundary) > 1e-15:
                # Gradient of (0.5 - |c_ik|) w.r.t c_ik is -sign(c_ik)
                grad_flat[i * 3 + min_dim] += w_boundary * (-np.sign(centers_reshaped[i, min_dim]))
        
        # We're minimizing -r, so gradient is -grad(r)
        return -r, -grad_flat
    
    # Run optimization
    try:
        result = minimize(
            objective_and_grad,
            centers.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < -best_radius:
            new_centers = result.x.reshape(N, 3)
            # Ensure points are within bounds
            new_centers = np.clip(new_centers, -0.5 + 1e-10, 0.5 - 1e-10)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > deadline:
        return best_centers
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() - start_time > deadline:
            break
        
        # Random initial guess
        centers_rand = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        try:
            result = minimize(
                objective_and_grad,
                centers_rand.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_radius:
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.5 + 1e-10, 0.5 - 1e-10)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_centers = new_centers
                    best_radius = new_radius
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
