"""
Hexagonal lattice packing strategy for 50 circles in a unit square.
Uses a parameterized hexagonal grid with variable row lengths to fit N=50.
Optimizes lattice parameters (spacing, offsets) using Differential Evolution,
then refines with gradient-based local search.
"""
import time
import numpy as np
import random
from scipy.optimize import differential_evolution, minimize

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _generate_hex_centers(params):
    """
    Generate centers from hexagonal lattice parameters.
    params: [dx, dy, phase_x, phase_y, row_offsets...]
    Structure: rows of 7, 6, 7, 6, 7, 6, 7, 4 circles.
    Total: 50 circles.
    
    We define a base hexagonal grid and allow slight deformations.
    To keep it simple and robust, we define the centers explicitly
    based on a few global parameters:
    - s: horizontal spacing
    - h: vertical spacing
    - shift_x: global x offset
    - shift_y: global y offset
    - row_shiftable: allow individual row x-shifts to be optimized? 
      For simplicity in DE, we'll optimize global params first, then local refine.
    
    Actually, let's parameterize the rows directly to allow "breathing".
    Rows:
    0: 7 circles
    1: 6 circles
    2: 7 circles
    3: 6 circles
    4: 7 circles
    5: 6 circles
    6: 7 circles
    7: 4 circles
    
    Hexagonal packing usually has rows offset by half spacing.
    Let's define the x-positions for each row based on a central alignment.
    """
    dx, dy, ox, oy = params[0], params[1], params[2], params[3]
    
    # Row definitions: (count, offset_factor)
    # Standard hex: even rows aligned, odd rows shifted by dx/2.
    # We can optimize the shift for each row type or just global.
    # Let's allow global dx, dy and two shift parameters: shift_even, shift_odd
    # But params length is fixed. Let's use:
    # p0: dx (x spacing)
    # p1: dy (y spacing)
    # p2: x_offset (global)
    # p3: y_offset (global)
    # p4: x_shift_even (extra x shift for even rows)
    # p5: x_shift_odd (extra x shift for odd rows)
    
    shift_even = params[4]
    shift_odd = params[5]
    
    rows_info = [
        (7, 0), # count, parity (0=even, 1=odd)
        (6, 1),
        (7, 0),
        (6, 1),
        (7, 0),
        (6, 1),
        (7, 0),
        (4, 1), # Last row is shorter, parity 1
    ]
    
    centers = []
    for i, (count, parity) in enumerate(rows_info):
        y = -0.5 + dy * (i + 0.5) + oy
        # Center the row horizontally
        # Width of row = (count - 1) * dx
        # Start x = -0.5 + dx/2 + ...
        # Let's center relative to 0
        row_width = (count - 1) * dx
        start_x = -row_width / 2.0
        
        # Apply specific shift for this parity
        if parity == 0:
            current_shift = shift_even
        else:
            current_shift = shift_odd
            
        for j in range(count):
            x = start_x + j * dx + ox + current_shift
            centers.append([x, y])
            
    return np.array(centers[:N])

def _objective_hex(params):
    """Negative radius for minimization."""
    centers = _generate_hex_centers(params)
    # Check validity
    if not np.all(np.isfinite(centers)):
        return 1.0
    if np.any(np.abs(centers) > 0.5 + 1e-9):
        return 1.0
    
    r = _compute_radius(centers)
    return -r

def _optimize_hex_structure(deadline):
    """Optimize the hexagonal lattice parameters using Differential Evolution."""
    # Bounds:
    # dx: 0.05 to 0.2
    # dy: 0.05 to 0.2
    # ox, oy: -0.1 to 0.1
    # shift_even, shift_odd: -0.1 to 0.1
    bounds = [
        (0.05, 0.2),   # dx
        (0.05, 0.2),   # dy
        (-0.1, 0.1),   # ox
        (-0.1, 0.1),   # oy
        (-0.1, 0.1),   # shift_even
        (-0.1, 0.1),   # shift_odd
    ]
    
    start_time = time.time()
    # DE can be slow, use a reasonable pop size and max iter
    # We have ~120s budget. Let's allocate ~60s to DE.
    de_deadline = start_time + 60.0
    
    try:
        result = differential_evolution(
            _objective_hex,
            bounds,
            seed=SEED,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            # Callback to check time
            callback=lambda xk, convergence: (time.time() > de_deadline)
        )
        
        if result.success or hasattr(result, 'x'):
            best_params = result.x
            best_centers = _generate_hex_centers(best_params)
            best_radius = -result.fun
            return best_centers, best_radius
    except Exception as e:
        pass
        
    # Fallback: simple grid
    return _initial_grid(), _compute_radius(_initial_grid())

def _initial_grid():
    """Fallback grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)

def _gradient_refinement(centers, deadline):
    """
    Refine the centers using gradient ascent on a soft-min surrogate.
    This allows individual circles to move off the strict lattice if it improves packing.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_time = time.time()
    lr = 0.005
    beta = 50.0
    
    def soft_min_radius(c, b):
        """Compute soft min radius and its gradient."""
        # Boundary
        bd = 0.5 - np.abs(c)
        max_bd = np.max(bd)
        w_bd = np.exp(-(bd - max_bd) * b)
        w_bd /= np.sum(w_bd)
        
        # Pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        pd = dists / 2.0
        
        # Only consider upper triangle for weights to avoid double counting in sum?
        # Soft min is symmetric, so weights should be symmetric.
        max_pd = np.max(pd[np.triu_indices_from(pd, k=1)])
        w_pd = np.exp(-(pd - max_pd) * b)
        np.fill_diagonal(w_pd, 0)
        w_pd /= np.sum(w_pd)
        
        # Combine
        # Objective is min(soft_min_boundary, soft_min_pair)
        # To make it differentiable, we can just sum the gradients weighted by which is active?
        # Or just use the soft min of the combined set of constraints?
        # Let's compute the value and gradient for both, then combine.
        
        val_bd = max_bd - np.log(np.sum(np.exp(-(bd - max_bd) * b))) / b
        val_pd = max_pd - np.log(np.sum(np.exp(-(pd - max_pd) * b))) / b
        
        # Gradient of soft min boundary
        grad_bd = np.zeros_like(c)
        for i in range(N):
            for k in range(2):
                if c[i, k] > 0:
                    grad_bd[i, k] -= w_bd[i, k]
                else:
                    grad_bd[i, k] += w_bd[i, k]
        
        # Gradient of soft min pairs
        grad_pd = np.zeros_like(c)
        # w_pd is symmetric.
        # d/dx_i sum_j w_ij * |x_i - x_j|/2
        # = sum_j w_ij * (x_i - x_j) / (2 * |x_i - x_j|)
        # Note: w_pd[i,j] is the weight.
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = dists[i, j]
                    if d > 1e-10:
                        factor = w_pd[i, j] / (2.0 * d)
                        grad_pd[i] += factor * diff[i, j]
                        
        # Combine gradients: if val_bd < val_pd, boundary is bottleneck, else pair is.
        # We can just add them with weights based on which is smaller?
        # A simpler approach: maximize the minimum of the two soft mins.
        # Let M = min(val_bd, val_pd).
        # If val_bd < val_pd, grad = grad_bd. Else grad = grad_pd.
        # This is non-smooth at the crossover.
        # Let's use a soft min of the two values themselves?
        # M_soft = soft_min(val_bd, val_pd, beta_high)
        
        # For simplicity and speed, let's just use the gradient of the dominant constraint.
        if val_bd < val_pd:
            return val_bd, grad_bd
        else:
            return val_pd, grad_pd
            
    for it in range(5000):
        if time.time() > deadline:
            break
            
        val, grad = soft_min_radius(best_centers, beta)
        
        # Normalize gradient
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-10:
            grad /= g_norm
            
        candidate = best_centers + lr * grad
        np.clip(candidate, -0.5, 0.5, out=candidate)
        
        new_radius = _compute_radius(candidate)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = candidate
            lr *= 1.05
        else:
            lr *= 0.9
            
        # Increase beta to sharpen
        if it % 100 == 0:
            beta *= 1.2
            
        if lr < 1e-8:
            break
            
    return best_centers, best_radius

def construct():
    """Main entry point."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    # 1. Optimize hexagonal lattice structure
    centers, radius = _optimize_hex_structure(deadline)
    
    # 2. Refine with gradient ascent
    if time.time() < deadline:
        remaining_deadline = deadline - 2.0 # Leave 2s for safety
        centers, radius = _gradient_refinement(centers, remaining_deadline)
        
    # Final check
    final_radius = _compute_radius(centers)
    if final_radius <= 0:
        centers = _initial_grid()
        
    return centers

# EVOLVE-BLOCK-END
