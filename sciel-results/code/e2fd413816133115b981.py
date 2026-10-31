"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    # Check inside square
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    # Check disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = np.maximum(x[2*n:], 1e-6)
    
    # Smooth penalty for boundary violations using soft-plus
    boundary_pen = 0.0
    for i in range(n):
        # Left boundary
        v = radii[i] - pos[i, 0]
        if v > 0:
            boundary_pen += v * v
        # Right boundary
        v = pos[i, 0] - (1.0 - radii[i])
        if v > 0:
            boundary_pen += v * v
        # Bottom boundary
        v = radii[i] - pos[i, 1]
        if v > 0:
            boundary_pen += v * v
        # Top boundary
        v = pos[i, 1] - (1.0 - radii[i])
        if v > 0:
            boundary_pen += v * v
    
    # Penalty for overlapping
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist)**2
    
    return -np.sum(radii) + 500.0 * (boundary_pen + overlap_pen)


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid: 25 circles
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.09
    
    # 26th circle: place in a corner
    pos[25, 0] = 0.1
    pos[25, 1] = 0.1
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.09
    # Row 0: 5 circles
    for i in range(5):
        idx = i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1
        radii[idx] = r
    
    # Row 1: 4 circles (offset)
    dy = r * np.sqrt(3)
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + dy
        radii[idx] = r
    
    # Row 2: 5 circles
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 2 * dy
        radii[idx] = r
    
    # Row 3: 4 circles (offset)
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + 3 * dy
        radii[idx] = r
    
    # Row 4: 5 circles
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 4 * dy
        radii[idx] = r
    
    # 26th circle
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.05
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively adjust positions and radii to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(500):
        valid = True
        
        # Check and fix boundaries by moving circles inward
        for i in range(N):
            # Left boundary
            if pos[i, 0] < radii[i]:
                pos[i, 0] = radii[i] + 1e-6
                valid = False
            # Right boundary
            if pos[i, 0] > 1.0 - radii[i]:
                pos[i, 0] = 1.0 - radii[i] - 1e-6
                valid = False
            # Bottom boundary
            if pos[i, 1] < radii[i]:
                pos[i, 1] = radii[i] + 1e-6
                valid = False
            # Top boundary
            if pos[i, 1] > 1.0 - radii[i]:
                pos[i, 1] = 1.0 - radii[i] - 1e-6
                valid = False
        
        # Check and fix overlaps by pushing apart
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-9:
                    # Push circles apart
                    if dist < 1e-12:
                        # Random push
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2 + 1e-4
                        pos[i, 0] += push * np.cos(angle)
                        pos[i, 1] += push * np.sin(angle)
                        pos[j, 0] -= push * np.cos(angle)
                        pos[j, 1] -= push * np.sin(angle)
                    else:
                        push = (min_dist - dist) / 2 + 1e-6
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] += push * nx
                        pos[i, 1] += push * ny
                        pos[j, 0] -= push * nx
                        pos[j, 1] -= push * ny
                    valid = False
        
        if valid:
            break
    
    # Final shrink pass to ensure strict validity
    for i in range(N):
        # Ensure inside square
        radii[i] = min(radii[i], pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(radii[i], 1e-4)
    
    for iteration in range(100):
        all_valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-9:
                    # Shrink both circles slightly
                    scale = dist / (min_dist + 1e-12) * 0.995
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    all_valid = False
        if all_valid:
            break
    
    return pos, radii


def _force_directed_relax(pos, radii, n_iter=2000, deadline=None):
    """Force-directed relaxation to find valid packing."""
    pos = pos.copy()
    radii = radii.copy()
    
    dt = 0.01
    damping = 0.9
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            # Left
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 10
            # Right
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i])) * 10
            # Bottom
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 10
            # Top
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i])) * 10
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        fx = np.cos(angle)
                        fy = np.sin(angle)
                    else:
                        fx = dx / dist
                        fy = dy / dist
                    overlap = min_dist - dist
                    force_mag = overlap * 5
                    forces[i, 0] += fx * force_mag
                    forces[i, 1] += fy * force_mag
                    forces[j, 0] -= fx * force_mag
                    forces[j, 1] -= fy * force_mag
        
        # Update positions
        pos += forces * dt
        # Clip to valid range
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], radii[i] + 1e-6, 1.0 - radii[i] - 1e-6)
            pos[i, 1] = np.clip(pos[i, 1], radii[i] + 1e-6, 1.0 - radii[i] - 1e-6)
    
    return pos, radii


def _grow_circles(pos, radii, deadline):
    """Iteratively grow circles while maintaining validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(1000):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try growing each circle slightly
        for i in range(N):
            # Check if we can grow circle i
            can_grow = True
            max_growth = 0.01
            
            # Boundary check
            max_growth = min(max_growth, pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                           pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            
            # Overlap check
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[i] - radii[j]
                if available < max_growth:
                    max_growth = available
            
            if max_growth > 1e-6:
                # Grow by a fraction of available space
                growth = max_growth * 0.5
                radii[i] += growth
                improved = True
        
        if not improved:
            break
    
    return pos, radii


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Step 1: Force-directed relaxation to get valid configuration
    if time.time() < deadline - 30:
        pos_fd, radii_fd = _force_directed_relax(pos, radii, n_iter=3000, deadline=deadline - 30)
        pos_pp, radii_pp = _post_process(pos_fd, radii_fd)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Step 2: Grow circles
    if time.time() < deadline - 20:
        pos_grow, radii_grow = _grow_circles(best_pos, best_radii, deadline=deadline - 20)
        pos_pp, radii_pp = _post_process(pos_grow, radii_grow)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Step 3: Gradient-based optimization with smooth objective
    if time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        bounds = []
        for i in range(N):
            bounds.append((1e-4, 1.0 - 1e-4))
            bounds.append((1e-4, 1.0 - 1e-4))
        for i in range(N):
            bounds.append((1e-4, 0.25))
        
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-12})
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Step 4: Final polish - more aggressive growth
    if time.time() < deadline - 5:
        pos_polish, radii_polish = _grow_circles(best_pos, best_radii, deadline=deadline - 5)
        pos_pp, radii_pp = _post_process(pos_polish, radii_polish)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Try different initial configurations
    init_configs = []
    
    # Grid
    pos, radii = _initial_grid()
    init_configs.append((pos, radii, 'grid'))
    
    # Hexagonal
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii, 'hex'))
    
    # Random with different seeds
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii, f'rand_{seed}'))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii, 'grid_pert'))
    
    # Run optimization from each initial config
    for pos, radii, name in init_configs:
        if time.time() > deadline - 15:
            break
        
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # If we didn't find a valid solution, fall back to simple grid
    if best_pos is None or best_score < 0:
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx, 0] = (i + 0.5) / side
                    pos[idx, 1] = (j + 0.5) / side
                    radii[idx] = 0.09
        # 26th circle
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
