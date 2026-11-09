"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Place 200 equal circles in a unit square [-0.5, 0.5]^2 to maximize the minimum radius.
    Uses L-BFGS-B to minimize an inverse-power-sum energy function that penalizes close pairs,
    with boundary constraints handled by the optimizer bounds.
    """
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # --- Initial guess: hexagonal packing pattern ---
    # For N=200, a hexagonal arrangement is a good starting point.
    # We'll create a hexagonal grid and then trim/pad to get exactly N points.
    
    def generate_hex_initial(N):
        """Generate initial positions using a hexagonal lattice pattern."""
        # Approximate density: 200 circles in area 1, so spacing ~ 1/sqrt(200) ≈ 0.0707
        # Hexagonal spacing: a = 2*r, where r is radius. Area per circle in hex packing = a^2 * sqrt(3)/2
        # So r ≈ 0.5 * a, and area per circle = 2*sqrt(3)*r^2
        # 200 * 2*sqrt(3)*r^2 = 1 => r ≈ 0.5/sqrt(400*sqrt(3)) ≈ 0.0366 (matches best known!)
        
        # Use a hexagonal lattice with spacing slightly larger than optimal to allow optimization
        # Let's start with a grid-based approach but with hexagonal offset for better initial spacing
        
        # Try different arrangements:
        # For 200 circles, we can use rows of alternating lengths
        # Hexagonal packing: rows alternate between n and n-1 circles
        
        # Let's try a rectangular grid first as baseline, then hex
        m = int(np.ceil(np.sqrt(N)))
        # Use a hexagonal-like pattern
        
        points = []
        # Calculate approximate spacing
        # For hexagonal packing with N points, we need roughly sqrt(N) rows
        # Let's use 15 rows for 200 points (15*13 = 195, 15*14 = 210)
        
        # Try to create a hexagonal pattern
        # Rows: 14, 13, 14, 13, ... alternating
        row_counts = []
        remaining = N
        for i in range(20):  # Max 20 rows
            if remaining <= 0:
                break
            # Alternate between two row sizes
            base = int(np.sqrt(N))  # ~14
            if i % 2 == 0:
                count = base
            else:
                count = base - 1
            if remaining <= count:
                count = remaining
            row_counts.append(count)
            remaining -= count
            if remaining <= 0:
                break
        
        # If we don't have enough, add more rows
        while remaining > 0:
            row_counts.append(min(remaining, int(np.sqrt(N))))
            remaining -= row_counts[-1]
        
        # Adjust to get exactly N points
        total = sum(row_counts)
        if total > N:
            # Remove excess from the end
            excess = total - N
            for i in range(len(row_counts) - 1, -1, -1):
                if excess <= 0:
                    break
                remove = min(excess, row_counts[i])
                row_counts[i] -= remove
                excess -= remove
            row_counts = [c for c in row_counts if c > 0]
        
        if total < N:
            # Add to first rows
            deficit = N - total
            for i in range(len(row_counts)):
                if deficit <= 0:
                    break
                row_counts[i] += deficit
                deficit -= row_counts[i]
        
        # Now generate positions
        # Hexagonal spacing
        a = 1.0 / np.sqrt(N * np.sqrt(3) / 2)  # Approximate hex spacing
        # Actually, let's just use a uniform spacing that fits
        spacing = 1.0 / (np.sqrt(N) + 0.5)  # Slightly smaller than 1/sqrt(N)
        
        y_pos = 0
        for i, count in enumerate(row_counts):
            if count <= 0:
                continue
            x_offset = (i % 2) * spacing / 2  # Hex offset
            for j in range(count):
                x = (j + 0.5) * spacing - 0.5 + x_offset
                y = (i + 0.5) * spacing - 0.5
                # Keep within bounds
                x = np.clip(x, -0.48, 0.48)
                y = np.clip(y, -0.48, 0.48)
                points.append([x, y])
        
        points = np.array(points[:N])
        return points
    
    def generate_grid_initial(N):
        """Generate initial positions using a square grid."""
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # --- Energy function: inverse power sum ---
    def energy_and_grad(x, p=8, eps=1e-8):
        """
        Compute energy = sum_{i<j} 1/d_ij^p + boundary_penalty
        and its gradient.
        """
        x = x.reshape(-1, 2)
        n = x.shape[0]
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1.0)  # Avoid division by zero
        d = np.sqrt(d2)
        
        # Inverse power sum
        # E_pair = sum_{i<j} 1/d_ij^p
        # Only upper triangle
        iu = np.triu_indices(n, k=1)
        d_ij = d[iu]
        E_pair = np.sum(d_ij**(-p))
        
        # Gradient of E_pair with respect to x
        # dE/dx_i = -p * sum_{j != i} (x_i - x_j) / d_ij^(p+2)
        # Compute using matrix operations
        d_safe = np.where(d > eps, d, eps)
        weights = d_safe**(-(p+2))
        np.fill_diagonal(weights, 0.0)
        
        # grad = -p * (diff * weights[..., np.newaxis]).sum(axis=1)
        # diff[i,j] = x_i - x_j
        grad_pair = -p * np.sum(diff * weights[:, :, np.newaxis], axis=1)
        
        # Boundary penalty: penalize points close to or outside [-0.5, 0.5]
        # Use a smooth penalty: for each coordinate, penalty = max(0, |x| - 0.49)^2 * large_weight
        # Or use tanh-based soft boundary
        # Let's use: penalty = sum of (max(0, |x_i| - 0.495))^2 * 1000
        boundary_margin = 0.495
        abs_x = np.abs(x)
        over = np.maximum(0, abs_x - boundary_margin)
        E_boundary = 1000.0 * np.sum(over**2)
        
        # Gradient of boundary penalty
        grad_boundary = np.zeros_like(x)
        mask = abs_x > boundary_margin
        grad_boundary[mask] = 2000.0 * over[mask] * np.sign(x[mask])
        
        E = E_pair + E_boundary
        grad = grad_pair + grad_boundary
        
        return E, grad.flatten()
    
    # --- Optimization ---
    best_x = None
    best_E = np.inf
    
    # Try multiple initializations
    initializations = []
    
    # 1. Grid
    initializations.append(generate_grid_initial(N))
    
    # 2. Hexagonal
    try:
        initializations.append(generate_hex_initial(N))
    except:
        pass
    
    # 3. Random with jitter
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        x_rand = rng.uniform(-0.45, 0.45, size=(N, 2))
        initializations.append(x_rand)
    
    for init_idx, x0 in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        x0_flat = x0.flatten()
        
        # L-BFGS-B with bounds
        bounds = [(-0.499, 0.499)] * (2 * N)
        
        try:
            result = minimize(
                energy_and_grad,
                x0_flat,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < best_E:
                best_x = result.x.reshape(-1, 2)
                best_E = result.fun
        except Exception as e:
            continue
    
    if best_x is None:
        # Fallback to grid
        best_x = generate_grid_initial(N)
    
    # --- Post-processing: local refinement using a different energy ---
    # Use a "soft minimum" based energy for final polishing
    def soft_min_energy(x, k=50, eps=1e-8):
        """
        Energy based on soft minimum of pairwise distances and boundary distances.
        E = -soft_min(d_ij) where soft_min uses log-sum-exp.
        """
        x = x.reshape(-1, 2)
        n = x.shape[0]
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        # Boundary distances
        # Distance to each wall
        d_top = 0.5 - x[:, 1]
        d_bottom = 0.5 + x[:, 1]
        d_right = 0.5 - x[:, 0]
        d_left = 0.5 + x[:, 0]
        boundary_d = np.minimum.reduce([d_top, d_bottom, d_right, d_left])
        boundary_d = np.maximum(boundary_d, eps)
        
        # Soft minimum of all distances (pairwise and boundary)
        # Combine all distances into one vector
        # Pairwise: upper triangle
        iu = np.triu_indices(n, k=1)
        d_pair = d[iu]
        d_pair = np.maximum(d_pair, eps)
        
        all_d = np.concatenate([d_pair, boundary_d])
        
        # Soft min: -log(sum(exp(-k*d)))/k
        # For numerical stability, subtract max
        min_d = np.min(all_d)
        exp_terms = np.exp(-k * (all_d - min_d))
        soft_min = min_d - np.log(np.sum(exp_terms)) / k
        
        E = -soft_min
        
        # Gradient
        # dE/dd = -d(soft_min)/dd
        # soft_min = min_d - log(sum(exp(-k*(all_d - min_d))))/k
        # This is complex; let's use a simpler approach
        
        # Alternative: use the inverse power sum with higher p for final polish
        return E
    
    # For simplicity, do a few more L-BFGS-B iterations with higher p
    if best_x is not None and time.time() < deadline - 5:
        x_flat = best_x.flatten()
        bounds = [(-0.499, 0.499)] * (2 * N)
        
        def energy_high_p(x, p=16, eps=1e-8):
            x = x.reshape(-1, 2)
            n = x.shape[0]
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            d2 = np.sum(diff**2, axis=2)
            np.fill_diagonal(d2, 1.0)
            d = np.sqrt(d2)
            
            iu = np.triu_indices(n, k=1)
            d_ij = d[iu]
            E_pair = np.sum(d_ij**(-p))
            
            d_safe = np.where(d > eps, d, eps)
            weights = d_safe**(-(p+2))
            np.fill_diagonal(weights, 0.0)
            grad_pair = -p * np.sum(diff * weights[:, :, np.newaxis], axis=1)
            
            boundary_margin = 0.499
            abs_x = np.abs(x)
            over = np.maximum(0, abs_x - boundary_margin)
            E_boundary = 10000.0 * np.sum(over**2)
            grad_boundary = np.zeros_like(x)
            mask = abs_x > boundary_margin
            grad_boundary[mask] = 20000.0 * over[mask] * np.sign(x[mask])
            
            E = E_pair + E_boundary
            grad = grad_pair + grad_boundary
            return E, grad.flatten()
        
        try:
            result = minimize(
                energy_high_p,
                x_flat,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.fun < best_E:
                best_x = result.x.reshape(-1, 2)
                best_E = result.fun
        except:
            pass
    
    # Final validation: ensure all points are within bounds
    best_x = np.clip(best_x, -0.4999, 0.4999)
    
    return best_x

# EVOLVE-BLOCK-END
