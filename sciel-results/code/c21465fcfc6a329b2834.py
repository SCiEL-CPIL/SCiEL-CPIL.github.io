"""Physics-based soft-body simulation for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _energy(pos, radii, W, H):
    """
    Compute total energy: repulsion between circles, wall repulsion, and a small
    regularization to keep radii positive and bounded.
    pos: (N, 2) array of centers
    radii: (N,) array of radii
    W, H: width and height of the rectangle
    """
    n = len(radii)
    energy = 0.0
    
    # Inter-circle repulsion (soft constraint for non-overlapping)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                # Strong repulsive potential when overlapping
                overlap = min_dist - dist
                energy += 1000.0 * overlap * overlap
    
    # Wall repulsion: circles must stay inside [0, W] x [0, H]
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        # Left wall
        if x < r:
            energy += 1000.0 * (r - x) ** 2
        # Right wall
        if x > W - r:
            energy += 1000.0 * (x - (W - r)) ** 2
        # Bottom wall
        if y < r:
            energy += 1000.0 * (r - y) ** 2
        # Top wall
        if y > H - r:
            energy += 1000.0 * (y - (H - r)) ** 2
    
    # Regularization: encourage larger radii (we maximize sum, so minimize negative sum)
    energy -= 10.0 * np.sum(radii)
    
    # Keep radii in reasonable bounds
    energy += 1.0 * np.sum((radii - 0.1) ** 2) * 0.01
    
    return energy


def _energy_grad(pos, radii, W, H):
    """Gradient of the energy function."""
    n = len(radii)
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)
    
    # Inter-circle gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                factor = 2000.0 * overlap / dist
                grad_pos[i, 0] += factor * dx
                grad_pos[i, 1] += factor * dy
                grad_pos[j, 0] -= factor * dx
                grad_pos[j, 1] -= factor * dy
                grad_radii[i] += 2000.0 * overlap
                grad_radii[j] += 2000.0 * overlap
    
    # Wall gradients
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        if x < r:
            g = 2000.0 * (r - x)
            grad_pos[i, 0] += -g
            grad_radii[i] += g
        if x > W - r:
            g = 2000.0 * (x - (W - r))
            grad_pos[i, 0] += g
            grad_radii[i] += -g
        if y < r:
            g = 2000.0 * (r - y)
            grad_pos[i, 1] += -g
            grad_radii[i] += g
        if y > H - r:
            g = 2000.0 * (y - (H - r))
            grad_pos[i, 1] += g
            grad_radii[i] += -g
    
    # Regularization gradients
    grad_radii -= 10.0
    grad_radii += 0.02 * (radii - 0.1)
    
    return grad_pos, grad_radii


def construct():
    # Perimeter = 2*(W + H) = 4 => W + H = 2
    # Try different aspect ratios
    best_result = None
    best_sum = 0.0
    
    # Aspect ratios to try: W/H
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
    
    for aspect in aspect_ratios:
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        # Initialize: grid-like placement with small jitter
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) * W / side + np.random.uniform(-0.01, 0.01)
            pos[i, 1] = (row + 0.5) * H / side + np.random.uniform(-0.01, 0.01)
        
        # Initial radii: small
        r_max = min(W, H) / (2 * side)
        radii = np.full(N, r_max * 0.5)
        
        # Optimize
        x0 = np.concatenate([pos.flatten(), radii])
        
        def objective(x):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            return _energy(p, r, W, H)
        
        def grad(x):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            gp, gr = _energy_grad(p, r, W, H)
            return np.concatenate([gp.flatten(), gr])
        
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-8})
        
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        
        # Clamp radii to be positive
        radii_opt = np.maximum(radii_opt, 1e-6)
        
        # Check validity: no overlaps, all inside
        valid = True
        for i in range(N):
            if pos_opt[i, 0] < radii_opt[i] or pos_opt[i, 0] > W - radii_opt[i]:
                valid = False
                break
            if pos_opt[i, 1] < radii_opt[i] or pos_opt[i, 1] > H - radii_opt[i]:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(pos_opt[i] - pos_opt[j])
                    if dist < radii_opt[i] + radii_opt[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            total = np.sum(radii_opt)
            if total > best_sum:
                best_sum = total
                best_result = (pos_opt, radii_opt, W, H)
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        W, H = 1.0, 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        return np.array(cells)
    
    pos_opt, radii_opt, W, H = best_result
    
    # Final safety clamp: ensure circles are strictly inside
    for i in range(N):
        radii_opt[i] = min(radii_opt[i], 
                           pos_opt[i, 0], 
                           W - pos_opt[i, 0], 
                           pos_opt[i, 1], 
                           H - pos_opt[i, 1])
        radii_opt[i] = max(radii_opt[i], 1e-6)
    
    # Resolve any remaining overlaps by shrinking
    for _ in range(100):
        resolved = True
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(pos_opt[i] - pos_opt[j])
                min_dist = radii_opt[i] + radii_opt[j]
                if dist < min_dist - 1e-10:
                    resolved = False
                    # Shrink both slightly
                    shrink = (min_dist - dist) / 2.0 + 1e-4
                    radii_opt[i] = max(radii_opt[i] - shrink, 1e-6)
                    radii_opt[j] = max(radii_opt[j] - shrink, 1e-6)
        if resolved:
            break
    
    # Ensure inside bounds after shrinking
    for i in range(N):
        radii_opt[i] = min(radii_opt[i], 
                           pos_opt[i, 0], 
                           W - pos_opt[i, 0], 
                           pos_opt[i, 1], 
                           H - pos_opt[i, 1])
        radii_opt[i] = max(radii_opt[i], 1e-6)
    
    result = np.column_stack([pos_opt, radii_opt])
    return result
# EVOLVE-BLOCK-END
